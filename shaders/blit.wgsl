struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) idx: u32) -> VSOut {
    var pos = array<vec2<f32>, 3>(
        vec2(-1.0, -3.0),
        vec2( 3.0,  1.0),
        vec2(-1.0,  1.0)
    );
    var uv = array<vec2<f32>, 3>(
        vec2(0.0, 2.0),
        vec2(2.0, 0.0),
        vec2(0.0, 0.0)
    );
    return VSOut(vec4(pos[idx], 0.0, 1.0), uv[idx]);
}

@group(0) @binding(0) var img:  texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
    return textureSample(img, samp, in.uv);
}
