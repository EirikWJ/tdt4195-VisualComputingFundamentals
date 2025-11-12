// Uncomment these following global attributes to silence most warnings of "low" interest:
/*
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unreachable_code)]
#![allow(unused_mut)]
#![allow(unused_unsafe)]
#![allow(unused_variables)]
*/
extern crate nalgebra_glm as glm;
use std::str::CharIndices;
use std::{ mem, ptr, os::raw::c_void };
use std::thread;
use std::sync::{Arc, Mutex, RwLock};

mod shader;
mod util;
mod mesh;
mod scene_graph;
use scene_graph::SceneNode;
mod toolbox;
use toolbox::simple_heading_animation;

use glutin::event::{Event, WindowEvent, DeviceEvent, KeyboardInput, ElementState::{Pressed, Released}, VirtualKeyCode::{self, *}};
use glutin::event_loop::ControlFlow;
use tobj::Model;
//use image::buffer;

// initial window size
const INITIAL_SCREEN_W: u32 = 1600;
const INITIAL_SCREEN_H: u32 = 800;

// == // Helper functions to make interacting with OpenGL a little bit prettier. You *WILL* need these! // == //

// Get the size of an arbitrary array of numbers measured in bytes
// Example usage:  byte_size_of_array(my_array)
fn byte_size_of_array<T>(val: &[T]) -> isize {
    std::mem::size_of_val(&val[..]) as isize
}

// Get the OpenGL-compatible pointer to an arbitrary array of numbers
// Example usage:  pointer_to_array(my_array)
fn pointer_to_array<T>(val: &[T]) -> *const c_void {
    &val[0] as *const T as *const c_void
}

// Get the size of the given type in bytes
// Example usage:  size_of::<u64>()
fn size_of<T>() -> i32 {
    mem::size_of::<T>() as i32
}

// Get an offset in bytes for n units of type T, represented as a relative pointer
// Example usage:  offset::<u64>(4)
fn offset<T>(n: u32) -> *const c_void {
    (n * mem::size_of::<T>() as u32) as *const T as *const c_void
}

// Get a null pointer (equivalent to an offset of 0)
// ptr::null()


// == // Generate your VAO here
unsafe fn create_vao(
    vertices: &Vec<f32>, 
    indices: &Vec<u32>, 
    colors: &Vec<f32>,
    normals: &Vec<f32>,
) -> u32 {
    // Implement me!

    // Also, feel free to delete comments :)

    // This should:
    // * Generate a VAO and bind it
    let mut vao: u32 = 0;
    gl::GenVertexArrays(1, &mut vao);
    gl::BindVertexArray(vao);

    // * Generate a VBO and bind it
    let mut vbo: u32 = 0;
    gl::GenBuffers(1, &mut vbo);
    gl::BindBuffer(gl::ARRAY_BUFFER, vbo);

    
    // * Fill it with data
    gl::BufferData(
        gl::ARRAY_BUFFER,  
        byte_size_of_array(vertices),
        pointer_to_array(vertices), 
        gl::STATIC_DRAW
    );
    // * Configure a VAP for the data and enable it
    let size = 3;
    gl::VertexAttribPointer(
        0, // 0 - 16
        size, 
        gl::FLOAT, 
        gl::FALSE, 
        size_of::<f32>()*size, 
        offset::<f32>(0)
    );
    gl::EnableVertexAttribArray(0);

    // ! Generate a CBO and bind it
    let mut cbo: u32 = 0;
    gl::GenBuffers(1, &mut cbo);
    gl::BindBuffer(gl::ARRAY_BUFFER, cbo);
    // ! Fill it with data
    gl::BufferData(
        gl::ARRAY_BUFFER,  
        byte_size_of_array(colors),
        pointer_to_array(colors), 
        gl::STATIC_DRAW
    );
    // ! Configure a VAP for the data and enable it
    let color_size = 4;
    gl::VertexAttribPointer(
        1, // 0 - 16
        color_size, 
        gl::FLOAT, 
        gl::FALSE, 
        size_of::<f32>()*color_size, 
        offset::<f32>(0)
    );
    gl::EnableVertexAttribArray(1);

    // * Generate a IBO and bind it
    let mut ibo = 0;
    gl::GenBuffers(1, &mut ibo);
    gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ibo);

    // * Fill it with data
    gl::BufferData(
        gl::ELEMENT_ARRAY_BUFFER, 
        byte_size_of_array(indices), 
        pointer_to_array(indices), 
        gl::STATIC_DRAW
    );

    // ? Generate a NBO and bind it
    let mut nbo: u32 = 0;
    gl::GenBuffers(1, &mut nbo);
    gl::BindBuffer(gl::ARRAY_BUFFER, nbo);
    // ? Fill it with data
    gl::BufferData(
        gl::ARRAY_BUFFER,  
        byte_size_of_array(normals),
        pointer_to_array(normals), 
        gl::STATIC_DRAW
    );
    // ? Configure a VAP for the data and enable it
    let normal_size = 3;
    gl::VertexAttribPointer(
        2, // 0 - 16
        normal_size, 
        gl::FLOAT, 
        gl::FALSE, 
        size_of::<f32>()*normal_size, 
        offset::<f32>(0)
    );
    gl::EnableVertexAttribArray(2);

    // * Return the ID of the VAO
    vao
}

unsafe fn draw_scene(
    node: &scene_graph::SceneNode,
    view_projection_matrix: &glm::Mat4,
    transformation_so_far: &glm::Mat4,
    shader: &shader::Shader
){
    // Perform any logic needed before drawing the node 
    let translation = glm::translation(&node.position);
    let reference_point = glm::translation(&(-node.reference_point));
    let object = glm::translation(&node.reference_point);
    
    let rotation_x = glm::rotation(
        node.rotation.x, &glm::vec3(1.0, 0.0, 0.0)
    );
    let rotation_y = glm::rotation(
        node.rotation.y, &glm::vec3(0.0, 1.0, 0.0)
    );
    let rotation_z = glm::rotation(
        node.rotation.z, &glm::vec3(0.0, 0.0, 1.0)
    );

    let local_transform = 
    translation * object * rotation_z * rotation_y * rotation_x *reference_point;

    let model = transformation_so_far * local_transform;
    let mvp = view_projection_matrix * model;
    gl::UniformMatrix4fv(
        shader.get_uniform_location("transform"),
        1,
        gl::FALSE,
        mvp.as_ptr(),
    );
    gl::UniformMatrix4fv(
    shader.get_uniform_location("model"),
    1,
    gl::FALSE,
    model.as_ptr(),
);
    
    // Check if node is drawable, if so: set uniforms, bind VAO and draw VAO 
    if node.index_count > 0 {
        gl::BindVertexArray(node.vao_id);
        gl::DrawElements(
            gl::TRIANGLES,
            node.index_count,
            gl::UNSIGNED_INT,
            std::ptr::null()
        );
    }
    // Recurse 
    for &child in &node.children { 
        draw_scene(
            &*child, 
            &mvp, 
            transformation_so_far, 
            shader); 
    }
}

fn main() {
    // Set up the necessary objects to deal with windows and event handling
    let el = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_title("Gloom-rs")
        .with_resizable(true)
        .with_inner_size(glutin::dpi::LogicalSize::new(INITIAL_SCREEN_W, INITIAL_SCREEN_H));
    let cb = glutin::ContextBuilder::new()
        .with_vsync(true);
    let windowed_context = cb.build_windowed(wb, &el).unwrap();
    // Uncomment these if you want to use the mouse for controls, but want it to be confined to the screen and/or invisible.
    // windowed_context.window().set_cursor_grab(true).expect("failed to grab cursor");
    // windowed_context.window().set_cursor_visible(false);

    // Set up a shared vector for keeping track of currently pressed keys
    let arc_pressed_keys = Arc::new(Mutex::new(Vec::<VirtualKeyCode>::with_capacity(10)));
    // Make a reference of this vector to send to the render thread
    let pressed_keys = Arc::clone(&arc_pressed_keys);

    // Set up shared tuple for tracking mouse movement between frames
    let arc_mouse_delta = Arc::new(Mutex::new((0f32, 0f32)));
    // Make a reference of this tuple to send to the render thread
    let mouse_delta = Arc::clone(&arc_mouse_delta);

    // Set up shared tuple for tracking changes to the window size
    let arc_window_size = Arc::new(Mutex::new((INITIAL_SCREEN_W, INITIAL_SCREEN_H, false)));
    // Make a reference of this tuple to send to the render thread
    let window_size = Arc::clone(&arc_window_size);

    // Spawn a separate thread for rendering, so event handling doesn't block rendering
    let render_thread = thread::spawn(move || {
        // Acquire the OpenGL Context and load the function pointers.
        // This has to be done inside of the rendering thread, because
        // an active OpenGL context cannot safely traverse a thread boundary
        let context = unsafe {
            let c = windowed_context.make_current().unwrap();
            gl::load_with(|symbol| c.get_proc_address(symbol) as *const _);
            c
        };

        let mut window_aspect_ratio = INITIAL_SCREEN_W as f32 / INITIAL_SCREEN_H as f32;

        // Set up openGL
        unsafe {
            gl::Enable(gl::DEPTH_TEST);
            gl::DepthFunc(gl::LESS);
            gl::Enable(gl::CULL_FACE);
            gl::Disable(gl::MULTISAMPLE);
            gl::Enable(gl::BLEND);
            gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
            gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS);
            gl::DebugMessageCallback(Some(util::debug_callback), ptr::null());

            // Print some diagnostics
            println!("{}: {}", util::get_gl_string(gl::VENDOR), util::get_gl_string(gl::RENDERER));
            println!("OpenGL\t: {}", util::get_gl_string(gl::VERSION));
            println!("GLSL\t: {}", util::get_gl_string(gl::SHADING_LANGUAGE_VERSION));
        }

        // == // Set up your VAO around here
        let lunar_path = "./resources/lunarsurface.obj";
        let lunar_surface = mesh::Terrain::load(&lunar_path);
        let lunar_surface_vao = unsafe {
            create_vao(
                &lunar_surface.vertices, 
                &lunar_surface.indices, 
                &lunar_surface.colors,
                &lunar_surface.normals
            )
        };

        let helicopter_path = "./resources/helicopter.obj";
        let helicopter_object = mesh::Helicopter::load(&helicopter_path);

        let heli_body_vao = unsafe {
            create_vao(
                &helicopter_object.body.vertices, 
                &helicopter_object.body.indices, 
                &helicopter_object.body.colors,
                &helicopter_object.body.normals
            )
        };
        let heli_door_vao = unsafe {
            create_vao(
                &helicopter_object.door.vertices, 
                &helicopter_object.door.indices, 
                &helicopter_object.door.colors,
                &helicopter_object.door.normals
            )
        };
        let heli_main_rotor_vao = unsafe {
            create_vao(
                &helicopter_object.main_rotor.vertices, 
                &helicopter_object.main_rotor.indices, 
                &helicopter_object.main_rotor.colors,
                &helicopter_object.main_rotor.normals
            )
        };
        let heli_tail_rotor_vao = unsafe {
            create_vao(
                &helicopter_object.tail_rotor.vertices, 
                &helicopter_object.tail_rotor.indices, 
                &helicopter_object.tail_rotor.colors,
                &helicopter_object.tail_rotor.normals
            )
        };

        let mut scene_graph_root = SceneNode::new();
        let mut lunar_surface_node = SceneNode::from_vao(
            lunar_surface_vao, 
            lunar_surface.index_count
        );
        lunar_surface_node.position = glm::vec3(0.0, 0.0, 0.0);
        scene_graph_root.add_child(&lunar_surface_node);

        let num_helicopters = 5;
        let mut helicopter_list: Vec<*mut SceneNode> = Vec::new();

        for _ in 0..num_helicopters {
            let mut helicopter_body_node = SceneNode::from_vao(
                heli_body_vao, 
                helicopter_object.body.index_count
            );
            helicopter_body_node.position = glm::vec3(0.0, 10.0, 0.0);
            
            let mut helicopter_door_node = SceneNode::from_vao(
                heli_door_vao, 
                helicopter_object.door.index_count
            );
            
            let mut helicopter_main_rotor_node = SceneNode::from_vao(
                heli_main_rotor_vao, 
                helicopter_object.main_rotor.index_count
            );
            helicopter_main_rotor_node.reference_point = glm::vec3(0.0, 0.0, 0.0);
            
            let mut helicopter_tail_rotor_node = SceneNode::from_vao(
                heli_tail_rotor_vao, 
                helicopter_object.tail_rotor.index_count
            );
            helicopter_tail_rotor_node.reference_point = glm::vec3(0.35, 2.3, 10.4);
            
            helicopter_body_node.add_child(&helicopter_door_node);
            helicopter_body_node.add_child(&helicopter_main_rotor_node);
            helicopter_body_node.add_child(&helicopter_tail_rotor_node);
            
            lunar_surface_node.add_child(&helicopter_body_node);
            
            // Get the last child we just added to lunar_surface_node
            let num_children = lunar_surface_node.n_children();
            let body_ptr = lunar_surface_node.children[num_children - 1];
            helicopter_list.push(body_ptr);
            
        }

        /*
                            lunar surface
                            helicopter body
                    door      main rotor       tail rotor
        */

        // == // Set up your shaders here

        // Basic usage of shader helper:
        // The example code below creates a 'shader' object.
        // It which contains the field `.program_id` and the method `.activate()`.
        // The `.` in the path is relative to `Cargo.toml`.
        // This snippet is not enough to do the exercise, and will need to be modified (outside
        // of just using the correct path), but it only needs to be called once

        
        let simple_shader = unsafe {
            shader::ShaderBuilder::new()
                .attach_file("./shaders/simple.vert")
                .attach_file("./shaders/simple.frag")
                .link()
        };
        
        // Used to demonstrate keyboard handling for exercise 2.
        let mut _arbitrary_number = 0.0; // feel free to remove


        // The main rendering loop
        let first_frame_time = std::time::Instant::now();
        let mut previous_frame_time = first_frame_time;
        
        //task 4 c) (a)
        let mut camera_position = glm::vec3(0.0f32, 0.0f32, 2.0f32);
        let mut camera_rotation = glm::vec3(0.0f32, 0.0f32, 0.0f32);

        loop {
            // Compute time passed since the previous frame and since the start of the program
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(first_frame_time).as_secs_f32();
            let delta_time = now.duration_since(previous_frame_time).as_secs_f32();
            previous_frame_time = now;

            // Handle resize events
            if let Ok(mut new_size) = window_size.lock() {
                if new_size.2 {
                    context.resize(glutin::dpi::PhysicalSize::new(new_size.0, new_size.1));
                    window_aspect_ratio = new_size.0 as f32 / new_size.1 as f32;
                    (*new_size).2 = false;
                    println!("Window was resized to {}x{}", new_size.0, new_size.1);
                    unsafe { gl::Viewport(0, 0, new_size.0 as i32, new_size.1 as i32); }
                }
            }
            
            let movement_speed =20.0; //how fast we're movin'
            let rotation_speed = 1.25; //how fast we're turnin'
            
            // Helicopter shenanigans for assignment 3
            
            let rotor_speed = 15.0; //how fast the rotors spin
            /*
            helicopter_main_rotor_node.rotation.y = rotor_speed * elapsed;
            helicopter_tail_rotor_node.rotation.x = rotor_speed * elapsed;
            
            let heading = simple_heading_animation(elapsed);
            helicopter_body_node.position.x = heading.x;
            helicopter_body_node.position.z = heading.z;

            helicopter_body_node.rotation.y = heading.yaw;
            helicopter_body_node.rotation.z = heading.pitch;
            helicopter_body_node.rotation.x = heading.roll;
            */
            for i in 0..num_helicopters {
                unsafe {
                    let time_offset = (i as f32) * 0.75;  // t seconds apart
                    let t = elapsed + time_offset;
                    
                    let heading = simple_heading_animation(t);
                    
                    let body = &mut *helicopter_list[i];
                    
                    body.position.x = heading.x;
                    body.position.z = heading.z;
                    
                    body.rotation.y = heading.yaw;
                    body.rotation.z = heading.pitch;
                    body.rotation.x = heading.roll;
            
                    body[1].rotation.y = rotor_speed * elapsed;
                    body[2].rotation.x = rotor_speed * elapsed;
                }
            }
            // forward vector based on current rotation (camera Z)
            let forward = glm::normalize(
                &glm::vec3(
                    camera_rotation.y.sin() * camera_rotation.x.cos(),
                    camera_rotation.x.sin(),
                    -camera_rotation.y.cos() * camera_rotation.x.cos(),
                )
            );
            // normal to the forward and up vectors (Camera X)
            let right = glm::normalize(
                &glm::cross(
                    &forward, 
                    &glm::vec3(0.0, 1.0, 0.0)
                )
            );

            // normal to the forward and right vectors (camara Y )
            let up    = glm::normalize( 
                &glm::cross(
                    &right,
                    &forward
                    )
                );

            // Handle keyboard input
            if let Ok(keys) = pressed_keys.lock() {
                for key in keys.iter() {
                    match key {
                        // The `VirtualKeyCode` enum is defined here:
                        //    https://docs.rs/winit/0.25.0/winit/event/enum.VirtualKeyCode.html

                        // Movement
                        VirtualKeyCode::W => {
                            camera_position += forward * delta_time * movement_speed;
                        }
                        VirtualKeyCode::S => {
                            camera_position -= forward * delta_time * movement_speed;
                        }
                        VirtualKeyCode::A => {
                            camera_position -= right * delta_time * movement_speed;
                        }
                        VirtualKeyCode::D => {
                            camera_position += right * delta_time * movement_speed;
                        }
                        VirtualKeyCode::Space  => {
                            camera_position += up * delta_time * movement_speed;
                        }
                        VirtualKeyCode::LShift => {
                            camera_position -= up * delta_time * movement_speed;
                        }
                        // Rotation
                        VirtualKeyCode::Up    => {
                            camera_rotation.x += delta_time * rotation_speed;
                        }
                        VirtualKeyCode::Down  => {
                            camera_rotation.x -= delta_time * rotation_speed;
                        }
                        VirtualKeyCode::Left  => {
                            camera_rotation.y -= delta_time * rotation_speed;
                        }
                        VirtualKeyCode::Right => {
                            camera_rotation.y += delta_time * rotation_speed;
                        }

                        _ => { }
                    }
                }
            }
            // Handle mouse movement. delta contains the x and y movement of the mouse since last frame in pixels
            if let Ok(mut delta) = mouse_delta.lock() {

                // == // Optionally access the accumulated mouse movement between
                // == // frames here with `delta.0` and `delta.1`

                camera_rotation.y += delta.0*0.005;
                camera_rotation.x -= delta.1*0.005;

                *delta = (0.0, 0.0); // reset when done
            }

            // == // Please compute camera transforms here (exercise 2 & 3)
            unsafe {
                simple_shader.activate();

                let model = glm::identity::<f32, 4>();

                let projection: glm::Mat4 = 
                    glm::perspective(
                        window_aspect_ratio,
                        0.8,
                        0.1,
                        100.0,
                    );

                let view = glm::look_at(
                    &camera_position, // where are we? at the camera xd
                    &(camera_position + forward), // what point are we looking at? forward...
                    &glm::vec3(0.0, 1.0, 0.0), // what direction is up? Y
                );

                let mvp = projection * view * model;

                gl::UniformMatrix4fv(
                    simple_shader.get_uniform_location("transform"),
                    1,
                    gl::FALSE,
                    mvp.as_ptr(),
                );


                gl::ClearColor(0.035, 0.046, 0.078, 1.0); // night sky
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
                draw_scene(
                    &scene_graph_root,
                    &mvp,
                    &model,
                    &simple_shader
                );
            }
            /*
            unsafe {
                // Clear the color and depth buffers
                gl::ClearColor(0.035, 0.046, 0.078, 1.0); // night sky
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);


                // == // Issue the necessary gl:: commands to draw your scene here

                    
                // Draw lunar terrain
                gl::BindVertexArray(lunar_surface_vao);
                gl::DrawElements(
                    gl::TRIANGLES,
                    lunar_surface.index_count,
                    gl::UNSIGNED_INT,
                    std::ptr::null()
                );
                
                // Draw helicopter body
                gl::BindVertexArray(heli_body_vao);
                gl::DrawElements(
                    gl::TRIANGLES,
                    helicopter_object.body.index_count,
                    gl::UNSIGNED_INT,
                    std::ptr::null()
                );
                
                // Draw helicopter door
                gl::BindVertexArray(heli_door_vao);
                gl::DrawElements(
                    gl::TRIANGLES,
                    helicopter_object.door.index_count,
                    gl::UNSIGNED_INT,
                    std::ptr::null()
                );
                
                // Draw main rotor
                gl::BindVertexArray(heli_main_rotor_vao);
                gl::DrawElements(
                    gl::TRIANGLES,
                    helicopter_object.main_rotor.index_count,
                    gl::UNSIGNED_INT,
                    std::ptr::null()
                );
                
                // Draw tail rotor
                gl::BindVertexArray(heli_tail_rotor_vao);
                gl::DrawElements(
                    gl::TRIANGLES,
                    helicopter_object.tail_rotor.index_count,
                    gl::UNSIGNED_INT,
                    std::ptr::null()
                );
                


            }
        */

            // Display the new color buffer on the display
            context.swap_buffers().unwrap(); // we use "double buffering" to avoid artifacts
        }
    });


    // == //
    // == // From here on down there are only internals.
    // == //


    // Keep track of the health of the rendering thread
    let render_thread_healthy = Arc::new(RwLock::new(true));
    let render_thread_watchdog = Arc::clone(&render_thread_healthy);
    thread::spawn(move || {
        if !render_thread.join().is_ok() {
            if let Ok(mut health) = render_thread_watchdog.write() {
                println!("Render thread panicked!");
                *health = false;
            }
        }
    });

    // Start the event loop -- This is where window events are initially handled
    el.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        // Terminate program if render thread panics
        if let Ok(health) = render_thread_healthy.read() {
            if *health == false {
                *control_flow = ControlFlow::Exit;
            }
        }

        match event {
            Event::WindowEvent { event: WindowEvent::Resized(physical_size), .. } => {
                println!("New window size received: {}x{}", physical_size.width, physical_size.height);
                if let Ok(mut new_size) = arc_window_size.lock() {
                    *new_size = (physical_size.width, physical_size.height, true);
                }
            }
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            }
            // Keep track of currently pressed keys to send to the rendering thread
            Event::WindowEvent { event: WindowEvent::KeyboardInput {
                    input: KeyboardInput { state: key_state, virtual_keycode: Some(keycode), .. }, .. }, .. } => {

                if let Ok(mut keys) = arc_pressed_keys.lock() {
                    match key_state {
                        Released => {
                            if keys.contains(&keycode) {
                                let i = keys.iter().position(|&k| k == keycode).unwrap();
                                keys.remove(i);
                            }
                        },
                        Pressed => {
                            if !keys.contains(&keycode) {
                                keys.push(keycode);
                            }
                        }
                    }
                }

                // Handle Escape and Q keys separately
                match keycode {
                    Escape => { *control_flow = ControlFlow::Exit; }
                    Q      => { *control_flow = ControlFlow::Exit; }
                    _      => { }
                }
            }
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                // Accumulate mouse movement
                if let Ok(mut position) = arc_mouse_delta.lock() {
                    *position = (position.0 + delta.0 as f32, position.1 + delta.1 as f32);
                }
            }
            _ => { }
        }
    });
}
