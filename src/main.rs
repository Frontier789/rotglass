#![feature(clamp)]
#![allow(dead_code)]
extern crate gl;
extern crate glui;
extern crate glui_proc;

use glui::graphics::{DrawShaderSelector, RenderCommand, RenderSequence};
use glui::gui::elements::SkipCell;
use glui::gui::widget::GuiPoint;
use glui::gui::*;
use glui::mecs::World;
use glui::mecs::*;
use glui::tools::*;
use std::f32::consts::PI;

#[derive(PartialEq, Clone, Debug)]
struct VmodelGui {
    line_system: SystemId,
    edit_mode: bool,
}

#[allow(unused_must_use)]
impl GuiBuilder for VmodelGui {
    fn build(&self) {
        let line_system = self.line_system;
        if self.edit_mode {
            -Lines {
                lines: vec![(
                    GuiPoint::relative(Vec2::new(0.5, 0.0)),
                    GuiPoint::relative(Vec2::new(0.5, 1.0)),
                )],
                color: Vec4::RED,
                ..Default::default()
            };
        }
        -GridLayout {
            col_widths: vec![GuiDimension::Default, GuiDimension::Units(110.0)],
            row_heights: vec![
                GuiDimension::Units(50.0),
                GuiDimension::Units(50.0),
                GuiDimension::Default,
            ],
            ..Default::default()
        } << {
            -SkipCell {};
            -Button {
                text: if self.edit_mode {
                    "rotate".to_string()
                } else {
                    "reset".to_string()
                },
                text_color: Vec4::BLACK,
                background: ButtonBckg::Fill(Vec4::WHITE.with_w(0.3)),
                callback: if self.edit_mode {
                    self.make_callback3(move |data, _, postbox| {
                        postbox.send(line_system, RotateBegin {});
                        data.edit_mode = false;
                    })
                } else {
                    self.make_callback3(move |data, _, postbox| {
                        postbox.send(line_system, RotateReset {});
                        data.edit_mode = true;
                    })
                },
                ..Default::default()
            };
            if !self.edit_mode {
                -SkipCell {};
                -Button {
                    text: "next shader".to_string(),
                    text_color: Vec4::BLACK,
                    background: ButtonBckg::Fill(Vec4::WHITE.with_w(0.3)),
                    callback: self.make_callback3(move |_, _, postbox| {
                        postbox.send(line_system, NextShader {});
                    }),
                    ..Default::default()
                };
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct RotateBegin {}
impl Message for RotateBegin {}

#[derive(Debug, Copy, Clone)]
struct RotateReset {}
impl Message for RotateReset {}

#[derive(Debug, Copy, Clone)]
struct NextShader {}
impl Message for NextShader {}

struct LineSystem {
    edit_mode: bool,
    screen_size: Vec2,
    camera_entity: Entity,
    points: Vec<Vec2>,
    line_draw: Entity,
    skybox: CubeTexture,
    current_shader: i32,
}

impl System for LineSystem {
    fn receive(&mut self, msg: &Box<dyn Message>, world: &mut StaticWorld) {
        if let Some(_) = msg.downcast_ref::<RotateBegin>() {
            let camera = world
                .component_mut::<DataComponent<Camera>>(self.camera_entity)
                .unwrap();
            let mut controller = ModelViewController::new(self.screen_size);
            controller.disable_roll = true;
            controller.spatial_mut().set_pos(Vec3::new(0.0, 0.0, 5.0));
            controller.spatial_mut().set_target(Vec3::origin());
            camera.data.set_controller(controller);

            self.points.pop();
            self.points.pop();
            if self.points.len() >= 2 {
                self.create_3d_draw(world);
            }
            self.edit_mode = false;
        }
        if let Some(_) = msg.downcast_ref::<RotateReset>() {
            let camera = world
                .component_mut::<DataComponent<Camera>>(self.camera_entity)
                .unwrap();
            camera.data.params.spatial.set_pos(Vec3::new(0.0, 0.0, 0.0));
            camera
                .data
                .params
                .spatial
                .set_target(Vec3::new(0.0, 0.0, -1.0));
            camera.data.set_controller(NoController {});

            self.update_draw_2d(world);

            self.points.clear();
            self.edit_mode = true;
        }
        if let Some(_) = msg.downcast_ref::<NextShader>() {
            let draw = world
                .component_mut::<DrawComponent>(self.line_draw)
                .unwrap();

            self.current_shader += 1;

            draw.render_seq.command_mut(0).shader = self.create_shader().into();
        }
    }
    fn window_event(&mut self, event: &GlutinWindowEvent, world: &mut StaticWorld) -> bool {
        if !self.edit_mode {
            return false;
        }

        match event {
            GlutinWindowEvent::Resized(s) => {
                self.screen_size = Vec2::new(s.width as f32, s.height as f32);
            }
            GlutinWindowEvent::CursorMoved { position, .. } => {
                let p = Vec2::new(position.x as f32, position.y as f32);

                self.set_last_point(
                    (p / self.screen_size.y * 2.0 - Vec2::new(self.screen_size.aspect(), 1.0))
                        * Vec2::new(1.0, -1.0),
                );
                self.update_draw_2d(world);
            }
            GlutinWindowEvent::MouseInput { state, button, .. } => {
                if *button == GlutinButton::Left && *state == GlutinElementState::Pressed {
                    self.duplicate_last_point();
                    self.update_draw_2d(world);
                }
            }
            _ => {}
        }

        false
    }
}

impl LineSystem {
    fn new(world: &mut World, camera_entity: Entity) -> LineSystem {
        let e = world.entity();

        world.add_component(
            e,
            DrawComponent {
                render_seq: RenderSequence::new(),
                model_matrix: Mat4::identity(),
            },
        );

        let skybox = CubeTexture::from_file("images/skymap_hi.png", FaceLayout::default()).unwrap();

        // for face in CubeFace::iter_faces() {
        //     skybox
        //         .face(*face)
        //         .as_image()
        //         .save(format!("skymap_{}.png", *face))
        //         .unwrap();
        // }

        LineSystem {
            edit_mode: true,
            camera_entity,
            screen_size: world.window_info().unwrap().size,
            points: vec![],
            line_draw: e,
            skybox,
            current_shader: 3,
        }
    }
    fn set_last_point(&mut self, p: Vec2) {
        match self.points.last_mut() {
            Some(last_p) => *last_p = p,
            None => self.points.push(p),
        }
    }
    fn duplicate_last_point(&mut self) {
        self.points
            .push(*self.points.last().unwrap_or(&Vec2::zero()));
    }
    fn update_draw_2d(&mut self, world: &mut StaticWorld) {
        let draw = world
            .component_mut::<DrawComponent>(self.line_draw)
            .unwrap();

        let pbuf = Buffer::from_vec(&self.points.clone());
        let mut vao = VertexArray::new();
        vao.attrib_buffer(0, &pbuf);

        let mut render_seq = RenderSequence::new();

        render_seq.add_buffer(pbuf.into_base_type());

        render_seq.add_command(RenderCommand::new_uniforms(
            vao,
            DrawMode::LineStrip,
            DrawShaderSelector::UniformColored,
            vec![Uniform::from("color", Vec4::WHITE)],
        ));

        draw.render_seq = render_seq;
        draw.model_matrix = Mat4::offset(Vec3::new(0.0, 0.0, -1.0));
    }

    fn create_3d_draw(&mut self, world: &mut StaticWorld) {
        let mut uniforms = vec![Uniform::from("skymap", &self.skybox)];

        for i in 1..self.points.len() {
            let cone = Vec4::from_vec2s(self.points[i - 1], self.points[i]);
            uniforms.push(Uniform::Vector4(format!("cones[{}]", i - 1), cone));
        }

        let pbuf = Buffer::from_vec(&vec![
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(-1.0, 1.0),
        ]);
        let mut vao = VertexArray::new();
        vao.attrib_buffer(0, &pbuf);

        let mut render_seq = RenderSequence::new();

        render_seq.add_buffer(pbuf.into_base_type());

        render_seq.add_command(RenderCommand::new_uniforms(
            vao,
            DrawMode::TriangleFan,
            self.create_shader().into(),
            uniforms,
        ));

        let draw = world
            .component_mut::<DrawComponent>(self.line_draw)
            .unwrap();
        draw.render_seq = render_seq;
        draw.model_matrix = Mat4::identity();
    }

    fn create_shader(&self) -> DrawShader {
        DrawShader::compile(
                    "#version 420 core
    
    layout(location = 0) in vec2 pos;

    uniform mat4 inv_view;
    uniform mat4 inv_projection;
    
    out vec3 ray_direction;

    void main()
    {
        vec4 view_ray = inv_projection * vec4(pos, 0, 1);
        vec4 world_ray = inv_view * vec4(view_ray.xyz, 0.0);
        ray_direction = world_ray.xyz;

        gl_Position = vec4(pos, 0, 1);
    }",
                    &("#version 420 core
    
    uniform vec3 cam_pos;
    uniform samplerCube skymap;
    
    in vec3 ray_direction;
    out vec4 clr;
    
    uniform vec4 cones[".to_owned() + &format!("{}", self.points.len() - 1) + "];
    
    bool trace(vec2 A, vec2 B, vec3 O, vec3 D, out float t, out vec3 n) {
        float ABy = A.y - B.y;
        float OBy = O.y - B.y;
        float a = ABy*ABy*D.x*D.x + ABy*ABy*D.z*D.z - A.x*A.x*D.y*D.y + 2*A.x*B.x*D.y*D.y - B.x*B.x*D.y*D.y;
        float b = 2*ABy*B.x*B.x*D.y + 2*ABy*ABy*D.x*O.x - 2*A.x*A.x*D.y*OBy + 2*ABy*ABy*D.z*O.z - 2*B.x*B.x*D.y*OBy - 2*ABy*A.x*B.x*D.y + 4*A.x*B.x*D.y*OBy;
        float c = -ABy*ABy*B.x*B.x + ABy*ABy*O.x*O.x + ABy*ABy*O.z*O.z - 2*ABy*A.x*B.x*OBy + 2*ABy*B.x*B.x*OBy - A.x*A.x*OBy*OBy + 2*A.x*B.x*OBy*OBy - B.x*B.x*OBy*OBy;
        float discriminant = b*b - 4*a*c;
        if (discriminant >= 0) {
            float sqrtd = sqrt(discriminant);
            float t1 = (-b + sqrtd) / (2*a);
            float t2 = (-b - sqrtd) / (2*a);
            float tf = min(t1,t2);
            float ts = max(t1,t2);
            
            if (tf > 0) {
                vec3 P = O + D*tf;
                if (P.y <= max(A.y, B.y) && P.y >= min(A.y, B.y)) {
                    t = tf;
                    float k = length(P.xz);
                    n = vec3(P.x/k*(A.y-B.y),(B.x-A.x),P.z/k*(A.y-B.y));
                    return true;
                }
            }
            if (ts > 0) {
                vec3 P = O + D*ts;
                if (P.y <= max(A.y, B.y) && P.y >= min(A.y, B.y)) {
                    t = ts;
                    float k = length(P.xz);
                    n = vec3(P.x/k*(A.y-B.y),(B.x-A.x),P.z/k*(A.y-B.y));
                    return true;
                }
            }
        }
        return false;
    }

    bool trace_all(vec3 O, vec3 D, out float t, out vec3 n) {
        t = 10000.0;
        n = vec3(0,0,0);
        bool hit = false; 
        
        for (int i=0;i<" + &format!("{}", self.points.len() - 1) + ";++i) {
            vec4 cone = cones[i];
            float tc;
            vec3  nc;
            if (trace(cone.xy, cone.zw, O, D, tc, nc)) 
            {
                if (tc < t) {
                    t = tc;
                    n = normalize(nc);
                    hit = true;
                }
            }
        }
        return hit;
    }
    
    float reflect_index(vec3 D, vec3 N, vec3 R, float eta) {
        float cosPhi_i = dot(N,-D);
        float cosPhi_t = dot(-N,R);
        float r_s = (cosPhi_i - eta * cosPhi_t) / (cosPhi_i + eta * cosPhi_t);
        return r_s * r_s;
    }

    struct Ray {
        vec3 O;
        vec3 D;
        float a;
    };
    
    void main()
    {" + 
        match self.current_shader % 5 {
            0 => "
        vec3 O = cam_pos;
        vec3 D = normalize(ray_direction);
        float t;
        vec3 N;
        if (trace_all(O, D, t, N)) {
            clr = vec4(1,1,1,1);
        } else { clr = vec4(0,0,0,1); }",
            1 => "
        vec3 O = cam_pos;
        vec3 D = normalize(ray_direction);
        float t;
        vec3 N;
        vec3 L = normalize(vec3(1,2,3));
        if (trace_all(O, D, t, N)) {
            clr = vec4(vec3(max(dot(L,N),0.1)),1);
            
            if (trace_all(O + D*t + N*0.001, L, t, N)) {
                clr = vec4(0.1,0.1,0.1,1);
            }
        } else { clr = vec4(vec3(pow(max(dot(L,D),0.1),13)),1); }",
            2 => "
        vec3 O = cam_pos;
        vec3 D = normalize(ray_direction);
        float red = 1; 
        float t;
        vec3 N;
        for (int i = 0; i < 20; ++i) {
            if (trace_all(O, D, t, N)) {
                if (dot(D,N) > 0) {N = -N;}
                O = O + D*t + N * 0.001;
                D = reflect(D,N);
                red = red * 0.95;
            } else {break;}
        }
        clr = texture(skymap, D) * vec4(red,red,red,1);",
            3 => "
        float eta0 = 1.1;
        vec3 O = cam_pos;
        vec3 D = normalize(ray_direction);
        float t;
        vec3 N;
        
        clr = vec4(0,0,0,0);

        for (int i=0; i<15; ++i) {
            if (trace_all(O, D, t, N)) {
                float eta = eta0;
                if (dot(D,N) > 0) {N = -N; eta = 1/eta0;} // lol
                
                vec3 On = (O + D*t) * 10;
                vec3 n = mix(vec3(sin(On.x),sin(On.y*1.1+1.3),sin(On.z*1.04+8.3)),vec3(0.0,0.0,1.0),0.9);
                vec3 u = cross(vec3(0,1,0),N);
                vec3 v = cross(N,u);
                N = n.x*u + n.y*v + n.z*N;

                vec3 R = refract(D,N,eta);

                if (R.x+R.y+R.z == 0) {
                    O = O + D*t + N*0.001;
                    D = reflect(D,N);
                } else {
                    O = O + D*t - N*0.001;
                    D = R;
                }
            } else {
                break;
            }
        }
        clr += texture(skymap,D);",
            4 => "
        Ray ray_stack[4];
        Ray next_ray_stack[4];
        int ray_count = 1;
        int next_ray = 0;
        
        float eta0 = 1.1;
        ray_stack[0] = Ray(cam_pos, normalize(ray_direction), 1);
        float t;
        vec3 N;
        
        clr = vec4(0,0,0,1);
        for (int i=0; i<3; ++i) {
            for (int j=0; j<ray_count; ++j) {
                vec3 O = ray_stack[j].O;
                vec3 D = ray_stack[j].D;
                float att = ray_stack[j].a;
                
                if (i < 2 && trace_all(O, D, t, N)) {
                    float eta = eta0;
                    if (dot(D,N) > 0) {N = -N; eta = 1/eta0;} // lol

                    vec3 R = refract(D,N,eta);

                    if (R.x+R.y+R.z == 0) {
                        next_ray_stack[next_ray] = Ray(O + D*t + N*0.001, reflect(D,N), att);
                        next_ray = next_ray + 1;
                    } else {
                        float r = reflect_index(D, N, R, eta) * 0.9;
                        
                        if (att * r > 0.001) {
                            next_ray_stack[next_ray] = Ray(O + D*t + N*0.001, reflect(D,N), att * r);
                            next_ray = next_ray + 1;
                        }

                        if (att * (1-r) > 0.001) {
                            next_ray_stack[next_ray] = Ray(O + D*t - N*0.001, R, att * (1-r));
                            next_ray = next_ray + 1;
                        }
                    }
                } else {
                    clr += texture(skymap, D) * att;
                }
            }
            ray_count = next_ray;
            next_ray = 0;
            ray_stack = next_ray_stack;
        }
        clr.w = 1;",
        _ => ""} + "}")).unwrap()
    }
}

fn main() {
    let mut w: World = World::new_win(Vec2::new(640.0, 480.0), "Vmodel", Vec3::grey(0.1));

    let ds = DrawSystem::new(&mut w, NoController {});
    let camid = ds.camera_entity;
    ds.camera_mut(w.as_static_mut()).params.fov = PI / 2.0;
    w.add_system(ds);

    let l = LineSystem::new(&mut w, camid);
    let id = w.add_system(l);

    w.add_gui(VmodelGui {
        line_system: id,
        edit_mode: true,
    });
    w.run();
}
