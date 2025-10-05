struct Camera {
    view_inv: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    params: vec4<f32>, // (width, height, time, _pad)
};

@group(0) @binding(0)
var outputTex: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(1)
var<uniform> camera: Camera;

fn world_ray_from_pixel(px: vec2<u32>) -> vec3<f32> {
    let dims = textureDimensions(outputTex);
    let uv = (vec2<f32>(px) + vec2<f32>(0.5, 0.5)) / vec2<f32>(f32(dims.x), f32(dims.y));
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0);
    let clip = vec4<f32>(ndc, 1.0);

    let view_pos = camera.proj_inv * clip;
    let view_pos3 = view_pos.xyz / view_pos.w;

    let world_pos = camera.view_inv * vec4<f32>(view_pos3, 1.0);
    let cam_pos   = (camera.view_inv * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;
    return normalize(world_pos.xyz - cam_pos);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(outputTex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let dir = world_ray_from_pixel(gid.xy);
    let rgb = 0.5 * (dir + vec3<f32>(1.0, 1.0, 1.0));
    textureStore(outputTex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(rgb, 1.0));
}
