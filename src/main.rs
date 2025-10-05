use std::f32::consts::PI;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3};
use wgpu::util::DeviceExt;

use winit::application::ApplicationHandler;
use winit::event::*;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes};
use winit::dpi::PhysicalSize;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUbo {
    view_inv: [[f32; 4]; 4],
    proj_inv: [[f32; 4]; 4],
    params: [f32; 4], // (width, height, time, _pad)
}

struct CameraCtrl {
    yaw: f32,
    pitch: f32,
    radius: f32,
    fov_y: f32,
    dragging: bool,
    last_cursor: Option<Vec2>,
}
impl CameraCtrl {
    fn new() -> Self {
        Self {
            yaw: 0.6,
            pitch: 0.3,
            radius: 4.0,
            fov_y: 60.0_f32.to_radians(),
            dragging: false,
            last_cursor: None,
        }
    }
    fn eye_target_up(&self) -> (Vec3, Vec3, Vec3) {
        let x = self.radius * self.yaw.cos() * self.pitch.cos();
        let y = self.radius * self.pitch.sin();
        let z = self.radius * self.yaw.sin() * self.pitch.cos();
        (Vec3::new(x, y, z), Vec3::ZERO, Vec3::Y)
    }
}

struct GpuState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,

    // compute output
    storage_tex: wgpu::Texture,
    storage_view: wgpu::TextureView,
    sampler: wgpu::Sampler,

    // camera
    camera_ctrl: CameraCtrl,
    camera_buf: wgpu::Buffer,

    // compute
    compute_bgl: wgpu::BindGroupLayout,
    compute_bg: wgpu::BindGroup,
    compute_pipeline: wgpu::ComputePipeline,

    // blit
    render_bgl: wgpu::BindGroupLayout,
    render_bg: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
}

impl GpuState {
    async fn new(
        instance: &wgpu::Instance,
        window: &Window,
        surface: &wgpu::Surface<'_>,
    ) -> Self {
        let size = window.inner_size();

        // Adapter / device
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(surface),
            })
            .await
            .expect("No GPU adapter");
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                }
            )
            .await
            .expect("device");

        // Surface config
        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Storage texture for compute
        let storage_format = wgpu::TextureFormat::Rgba8Unorm;
        let (storage_tex, storage_view) =
            create_storage_texture(&device, config.width, config.height, storage_format);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Shaders
        let trace_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("trace.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/trace.wgsl").into()),
        });
        let blit_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/blit.wgsl").into()),
        });

        // Camera UBO
        let camera_ctrl = CameraCtrl::new();
        let (view_inv, proj_inv) = compute_camera_mats(&camera_ctrl, config.width, config.height);
        let ubo = CameraUbo {
            view_inv: view_inv.to_cols_array_2d(),
            proj_inv: proj_inv.to_cols_array_2d(),
            params: [config.width as f32, config.height as f32, 0.0, 0.0],
        };
        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera_ubo"),
            contents: bytemuck::bytes_of(&ubo),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Compute pipeline
        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: storage_format,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let compute_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("compute_pl"),
            bind_group_layouts: &[&compute_bgl],
            push_constant_ranges: &[],
        });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("trace_compute"),
            layout: Some(&compute_pl),
            module: &trace_module,
            entry_point: Some("main"),
            cache: None,
            compilation_options: Default::default(),
        });
        let compute_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bg"),
            layout: &compute_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&storage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: camera_buf.as_entire_binding(),
                },
            ],
        });

        // Render pipeline (fullscreen triangle)
        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let render_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render_pl"),
            bind_group_layouts: &[&render_bgl],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit_pipeline"),
            layout: Some(&render_pl),
            vertex: wgpu::VertexState {
                module: &blit_module,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_module,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let render_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_bg"),
            layout: &render_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&storage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self {
            device,
            queue,
            config,
            size,
            storage_tex,
            storage_view,
            sampler,
            camera_ctrl,
            camera_buf,
            compute_bgl,
            compute_bg,
            compute_pipeline,
            render_bgl,
            render_bg,
            render_pipeline,
        }
    }

    fn resize(&mut self, surface: &wgpu::Surface<'_>, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        surface.configure(&self.device, &self.config);

        let (tex, view) = create_storage_texture(
            &self.device,
            self.config.width,
            self.config.height,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        self.storage_tex = tex;
        self.storage_view = view;

        self.compute_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bg"),
            layout: &self.compute_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.storage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.camera_buf.as_entire_binding(),
                },
            ],
        });
        self.render_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render_bg"),
            layout: &self.render_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.storage_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        self.update_camera_buffer(0.0);
    }

    fn update_camera_buffer(&mut self, time: f32) {
        let (view_inv, proj_inv) =
            compute_camera_mats(&self.camera_ctrl, self.config.width, self.config.height);
        let ubo = CameraUbo {
            view_inv: view_inv.to_cols_array_2d(),
            proj_inv: proj_inv.to_cols_array_2d(),
            params: [self.config.width as f32, self.config.height as f32, time, 0.0],
        };
        self.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&ubo));
    }

    fn render(&mut self, surface: &wgpu::Surface<'_>, time: f32) -> Result<(), wgpu::SurfaceError> {
        self.update_camera_buffer(time);

        let frame = surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // compute
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("trace_compute"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.compute_bg, &[]);
            let wg_x = (self.size.width + 7) / 8;
            let wg_y = (self.size.height + 7) / 8;
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // blit
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit_render"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.render_bg, &[]);
            rpass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}

fn create_storage_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("storage_tex"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

fn compute_camera_mats(ctrl: &CameraCtrl, width: u32, height: u32) -> (Mat4, Mat4) {
    let (eye, target, up) = ctrl.eye_target_up();
    let view = Mat4::look_at_rh(eye, target, up);
    let view_inv = view.inverse();

    let aspect = (width.max(1) as f32) / (height.max(1) as f32);
    let proj = Mat4::perspective_rh(ctrl.fov_y, aspect, 0.1, 1000.0);
    let proj_inv = proj.inverse();
    (view_inv, proj_inv)
}

// ---------- App / ApplicationHandler ----------
struct App {
    instance: wgpu::Instance,
    window:  Option<&'static Window>,
    surface: Option<wgpu::Surface<'static>>,
    state:   Option<GpuState>,
    start:   Instant,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, elwt: &ActiveEventLoop) {
        if self.window.is_none() {
            // Create the window
            let win = elwt
                .create_window(WindowAttributes::default().with_title("Black Hole — wgpu27 / winit30"))
                .expect("window");

            // Leak to get &'static Window (ok for a single-window app)
            let win_static: &'static Window = Box::leak(Box::new(win));
            self.window = Some(win_static);

            // Create surface borrowing the stored window
            let surf = self.instance.create_surface(win_static).expect("surface");
            self.surface = Some(surf);

            // Build GPU state
            let st = pollster::block_on(GpuState::new(
                &self.instance,
                win_static,
                self.surface.as_ref().unwrap(),
            ));
            self.state = Some(st);
            self.start = Instant::now();
        }
    }

    fn window_event(
        &mut self,
        elwt: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if let (Some(win), Some(surf), Some(st)) =
            (self.window, self.surface.as_ref(), self.state.as_mut())
        {
            if window_id != win.id() { return; }

            match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(new_size) => st.resize(surf, new_size),

                WindowEvent::MouseInput { state: mstate, button: MouseButton::Left, .. } => {
                    st.camera_ctrl.dragging = mstate == ElementState::Pressed;
                    if !st.camera_ctrl.dragging { st.camera_ctrl.last_cursor = None; }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if st.camera_ctrl.dragging {
                        let pos = Vec2::new(position.x as f32, position.y as f32);
                        if let Some(prev) = st.camera_ctrl.last_cursor {
                            let delta = pos - prev;
                            let sensitivity = 0.005;
                            st.camera_ctrl.yaw   -= delta.x * sensitivity;
                            st.camera_ctrl.pitch -= delta.y * sensitivity;
                            let limit = 0.995 * (PI / 2.0);
                            st.camera_ctrl.pitch = st.camera_ctrl.pitch.clamp(-limit, limit);
                            win.request_redraw();
                        }
                        st.camera_ctrl.last_cursor = Some(pos);
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let scroll = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(p) => (p.y as f32 / 50.0) as f32,
                    };
                    let factor = (1.0 - scroll * 0.1).clamp(0.2, 5.0);
                    st.camera_ctrl.radius = (st.camera_ctrl.radius * factor).clamp(1.0, 50.0);
                    win.request_redraw();
                }
                WindowEvent::RedrawRequested => {
                    let t = self.start.elapsed().as_secs_f32();
                    if let Err(e) = st.render(surf, t) {
                        match e {
                            wgpu::SurfaceError::Lost => st.resize(surf, st.size),
                            wgpu::SurfaceError::OutOfMemory => elwt.exit(),
                            _ => eprintln!("{e:?}"),
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn about_to_wait(&mut self, _elwt: &ActiveEventLoop) {
        if let Some(win) = self.window {
            win.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("event loop");
    let mut app = App {
        instance: wgpu::Instance::new(&wgpu::InstanceDescriptor::default()),
        window: None,
        surface: None,
        state: None,
        start: Instant::now(),
    };
    event_loop.run_app(&mut app).expect("run_app");
}
