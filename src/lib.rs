use futures::channel::oneshot;

static USE_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

pub struct App {
    #[allow(dead_code)]
    instance: wgpu::Instance,
    #[allow(dead_code)]
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
}

impl std::fmt::Display for App {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "App_backend:{:?}", self.adapter.get_info().backend)
    }
}

impl App {
    pub async fn new(backend: Option<wgpu::Backends>) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: backend.unwrap_or(wgpu::Backends::all()),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&Default::default(), None)
            .await
            .unwrap();

        let pipeline = Self::build_pipeline(&device);

        Self {
            instance,
            adapter,
            device,
            queue,
            pipeline,
        }
    }

    fn build_pipeline(device: &wgpu::Device) -> wgpu::RenderPipeline {
        let vs_src = include_str!("shader.vert");
        let fs_src = include_str!("shader.frag");
        let compiler = shaderc::Compiler::new().unwrap();
        let vs_spirv = compiler
            .compile_into_spirv(
                vs_src,
                shaderc::ShaderKind::Vertex,
                "shader.vert",
                "main",
                None,
            )
            .unwrap();
        let fs_spirv = compiler
            .compile_into_spirv(
                fs_src,
                shaderc::ShaderKind::Fragment,
                "shader.frag",
                "main",
                None,
            )
            .unwrap();
        let vs_data = wgpu::util::make_spirv(vs_spirv.as_binary_u8());
        let fs_data = wgpu::util::make_spirv(fs_spirv.as_binary_u8());
        let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex Shader"),
            source: vs_data,
        });
        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fragment Shader"),
            source: fs_data,
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: "main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: "main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: USE_TEXTURE_FORMAT,
                    blend: Some(wgpu::BlendState {
                        alpha: wgpu::BlendComponent::REPLACE,
                        color: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            // If the pipeline will be used with a multiview render pass, this
            // indicates how many array layers the attachments will have.
            multiview: None,
            cache: None,
        });
        render_pipeline
    }

    pub fn create_render_buffer(&self, texture_size: (u32, u32)) -> (wgpu::Buffer, wgpu::Extent3d) {
        let device = &self.device;
        let queue = &self.queue;

        let texture_desc = wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: texture_size.0,
                height: texture_size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: None,
            view_formats: &[],
        };
        let texture = device.create_texture(&texture_desc);
        let texture_view = texture.create_view(&Default::default());

        // we need to store this for later
        let u32_size = std::mem::size_of::<u32>() as u32;

        let output_buffer_size =
            (u32_size * texture_size.0 * texture_size.1) as wgpu::BufferAddress;
        let output_buffer_desc = wgpu::BufferDescriptor {
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST
                // this tells wpgu that we want to read this buffer from the cpu
                | wgpu::BufferUsages::MAP_READ,
            label: None,
            mapped_at_creation: false,
        };
        let output_buffer = device.create_buffer(&output_buffer_desc);

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let render_pass_desc = wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            };
            let mut render_pass = encoder.begin_render_pass(&render_pass_desc);

            render_pass.set_pipeline(&self.pipeline);
            render_pass.draw(0..3, 0..1);
        }

        // let ts = std::time::Instant::now();

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(u32_size * texture_size.0),
                    rows_per_image: Some(texture_size.1),
                },
            },
            texture_desc.size,
        );

        queue.submit(Some(encoder.finish()));
        // let time = ts.elapsed();
        // println!("copy_texture_to_buffer time: {:?}", time);

        (output_buffer, texture_desc.size)
    }

    pub async fn render_image(&self, texture_size: (u32, u32), output_buffer: wgpu::Buffer) {
        // We need to scope the mapping variables so that we can
        // unmap the buffer
        let ts = std::time::Instant::now();
        {
            let buffer_slice = output_buffer.slice(..);

            // NOTE: We have to create the mapping THEN device.poll() before await
            // the future. Otherwise the application will freeze.
            let (tx, rx) = oneshot::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            self.device.poll(wgpu::Maintain::Wait);
            rx.await.unwrap().unwrap();

            println!("map_async time: {:?}", ts.elapsed());
            let ts = std::time::Instant::now();

            let data = buffer_slice.get_mapped_range();

            use image::{ImageBuffer, Rgba};
            let buffer =
                ImageBuffer::<Rgba<u8>, _>::from_raw(texture_size.0, texture_size.1, data).unwrap();
            let image_name = format!("image_{}_{}.png", texture_size.0, texture_size.1);
            buffer.save(image_name).unwrap();

            println!("save image time: {:?}", ts.elapsed());
        }
        output_buffer.unmap();
    }

    pub async fn just_map_buffer(&self, output_buffer: wgpu::Buffer) -> Vec<u8> {
        let buffer_slice = output_buffer.slice(..);

        // NOTE: We have to create the mapping THEN device.poll() before await
        // the future. Otherwise the application will freeze.
        let (tx, rx) = oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        // return first 10 bytes
        let result = data[..10].to_vec();

        drop(data);
        output_buffer.unmap();

        result
    }

    pub async fn extract_buffer_data(
        &self,
        texture_size: (u32, u32),
        output_buffer: &wgpu::Buffer,
    ) -> RawBuffer {
        let ts = std::time::Instant::now();
        let buffer_slice = output_buffer.slice(..);

        // NOTE: We have to create the mapping THEN device.poll() before await
        // the future. Otherwise the application will freeze.
        let (tx, rx) = oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result = RawBuffer::new(data.to_vec(), texture_size.0, texture_size.1, 4);

        drop(data);
        output_buffer.unmap();

        println!("extract_buffer_data time: {:?}", ts.elapsed());

        result
    }

    pub async fn generate_fill_mask(
        &self,
        texture_size: (u32, u32),
        seed_pos: (u32, u32),
        tolerance: f32,
        output_buffer: &wgpu::Buffer,
    ) -> RawBuffer {
        let ts = std::time::Instant::now();
        let buffer_slice = output_buffer.slice(..);

        // NOTE: We have to create the mapping THEN device.poll() before await
        // the future. Otherwise the application will freeze.
        let (tx, rx) = oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.await.unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let img_buffer = ImgBufferRef::new(data.as_ref(), texture_size.0, texture_size.1, 4);
        let mask = img_buffer.flood_fill_to_mask(seed_pos.0, seed_pos.1, tolerance);

        drop(data);
        output_buffer.unmap();

        println!("generate_fill_mask time: {:?}", ts.elapsed());

        mask
    }
}
use image::{ImageBuffer, Luma, Pixel, Rgba};

struct ImgBufferRef<'a> {
    data: &'a [u8],
    width: u32,
    height: u32,
    pixel_stride: u32,
}

impl<'a> ImgBufferRef<'a> {
    pub fn new(data: &'a [u8], width: u32, height: u32, pixel_stride: u32) -> Self {
        Self {
            data,
            width,
            height,
            pixel_stride,
        }
    }

    fn get_pixel(&self, x: u32, y: u32) -> Rgba<u8> {
        let index = (y * self.width + x) as usize * self.pixel_stride as usize;
        *Rgba::from_slice(&self.data[index..index + self.pixel_stride as usize])
    }

    fn flood_fill_to_mask(&self, x: u32, y: u32, fill_boundary_tolerance: f32) -> RawBuffer {
        let mut mask = vec![0u8; self.width as usize * self.height as usize];
        let original_color = self.get_pixel(x, y);

        let mut stack = vec![(x, y)];
        let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];

        let mask_value = 255u8;

        while let Some((cx, cy)) = stack.pop() {
            if !self.colors_within_tolerance(
                self.get_pixel(cx, cy),
                original_color,
                fill_boundary_tolerance,
            ) {
                continue;
            }

            let index = (cy * self.width + cx) as usize;
            if mask[index] == mask_value {
                continue;
            }

            mask[index] = mask_value;

            for (dx, dy) in directions.iter() {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;
                if nx >= 0 && nx < self.width as i32 && ny >= 0 && ny < self.height as i32 {
                    stack.push((nx as u32, ny as u32));
                }
            }
        }

        RawBuffer::new(mask, self.width, self.height, 1)
    }

    fn colors_within_tolerance(&self, color1: Rgba<u8>, color2: Rgba<u8>, tolerance: f32) -> bool {
        let diff = color1
            .0
            .iter()
            .zip(color2.0.iter())
            .map(|(a, b)| (*a as f32 - *b as f32).abs())
            .sum::<f32>();
        diff / 4.0 <= tolerance * 255.0
    }
}

#[derive(Clone)]
pub struct RawBuffer {
    pub data: Vec<u8>,
    width: u32,
    height: u32,
    #[allow(dead_code)]
    pixel_stride: u32,
}

impl RawBuffer {
    pub fn new(data: Vec<u8>, width: u32, height: u32, pixel_stride: u32) -> Self {
        Self {
            data,
            width,
            height,
            pixel_stride,
        }
    }

    // flood fill algorithm
    // tolerance is the maximum difference between the original color and the target color.
    // the bigger the tolerance, the more colors will be filled.
    // range of 0.0 to 1.0
    pub fn flood_fill(
        &mut self,
        x: u32,
        y: u32,
        target_color: Rgba<u8>,
        fill_boundary_tolerance: f32,
    ) {
        let original_color = self.get_pixel(x, y);
        if self.colors_within_tolerance(original_color, target_color, fill_boundary_tolerance) {
            return;
        }

        let mut stack = vec![(x, y)];
        let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];

        while let Some((cx, cy)) = stack.pop() {
            if !self.colors_within_tolerance(
                self.get_pixel(cx, cy),
                original_color,
                fill_boundary_tolerance,
            ) {
                continue;
            }

            self.set_pixel(cx, cy, target_color);

            for (dx, dy) in directions.iter() {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;
                if nx >= 0 && nx < self.width as i32 && ny >= 0 && ny < self.height as i32 {
                    stack.push((nx as u32, ny as u32));
                }
            }
        }
    }

    pub fn flood_fill_to_mask(&self, x: u32, y: u32, fill_boundary_tolerance: f32) -> RawBuffer {
        let mut mask = vec![0u8; self.width as usize * self.height as usize];
        let original_color = self.get_pixel(x, y);

        let mut stack = vec![(x, y)];
        let directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];

        let mask_value = 255u8;

        while let Some((cx, cy)) = stack.pop() {
            if !self.colors_within_tolerance(
                self.get_pixel(cx, cy),
                original_color,
                fill_boundary_tolerance,
            ) {
                continue;
            }

            let index = (cy * self.width + cx) as usize;
            if mask[index] == mask_value {
                continue;
            }

            mask[index] = mask_value;

            for (dx, dy) in directions.iter() {
                let nx = cx as i32 + dx;
                let ny = cy as i32 + dy;
                if nx >= 0 && nx < self.width as i32 && ny >= 0 && ny < self.height as i32 {
                    stack.push((nx as u32, ny as u32));
                }
            }
        }

        RawBuffer::new(mask, self.width, self.height, 1)
    }

    /// flood fill algorithm using scanline technique
    /// x,y is the seed position
    pub fn flood_fill_to_mask_scanline(&self, x: u32, y: u32, fill_boundary_tolerance: f32) -> RawBuffer {
        let mut mask = vec![0u8; self.width as usize * self.height as usize];
        let original_color = self.get_pixel(x, y);
    
        let mask_value = 255u8;
        let mut stack = vec![(x, y)];
    
        while let Some((x, y)) = stack.pop() {
            let mut x1 = x as i32;
            // Skip if already filled
            if mask[(y * self.width + x as u32) as usize] == mask_value {
                continue;
            }
            while x1 >= 0 && self.colors_within_tolerance(self.get_pixel(x1 as u32, y), original_color, fill_boundary_tolerance) {
                x1 -= 1;
            }
            x1 += 1;
    
            let mut span_above = false;
            let mut span_below = false;
    
            while x1 < self.width as i32 && self.colors_within_tolerance(self.get_pixel(x1 as u32, y), original_color, fill_boundary_tolerance) {
                let index = (y * self.width + x1 as u32) as usize;
                
                // Skip if already filled
                if mask[index] == mask_value {
                    x1 += 1;
                    continue;
                }

                mask[index] = mask_value;

                if !span_above && y > 0 && self.colors_within_tolerance(self.get_pixel(x1 as u32, y - 1), original_color, fill_boundary_tolerance) {
                    stack.push((x1 as u32, y - 1));
                    span_above = true;
                } else if span_above && y > 0 && !self.colors_within_tolerance(self.get_pixel(x1 as u32, y - 1), original_color, fill_boundary_tolerance) {
                    span_above = false;
                }

                if !span_below && y < self.height - 1 && self.colors_within_tolerance(self.get_pixel(x1 as u32, y + 1), original_color, fill_boundary_tolerance) {
                    stack.push((x1 as u32, y + 1));
                    span_below = true;
                } else if span_below && y < self.height - 1 && !self.colors_within_tolerance(self.get_pixel(x1 as u32, y + 1), original_color, fill_boundary_tolerance) {
                    span_below = false;
                }

                x1 += 1;
            }
        }
    
        RawBuffer::new(mask, self.width, self.height, 1)
    }

    fn colors_within_tolerance(&self, color1: Rgba<u8>, color2: Rgba<u8>, tolerance: f32) -> bool {
        let diff = color1
            .0
            .iter()
            .zip(color2.0.iter())
            .map(|(a, b)| (*a as f32 - *b as f32).abs())
            .sum::<f32>();
        diff / 4.0 <= tolerance * 255.0
    }

    fn get_pixel(&self, x: u32, y: u32) -> Rgba<u8> {
        let index = (y * self.width + x) as usize * 4;
        *Rgba::from_slice(&self.data[index..index + 4])
    }

    fn set_pixel(&mut self, x: u32, y: u32, color: Rgba<u8>) {
        let index = (y * self.width + x) as usize * 4;
        self.data[index..index + 4].copy_from_slice(&color.0);
    }

    pub fn save_as_image(&self, name: &str) {
        match self.pixel_stride {
            4 => {
                let buffer = ImageBuffer::<Rgba<u8>, _>::from_raw(
                    self.width,
                    self.height,
                    self.data.clone(),
                )
                .unwrap();
                buffer.save(name).unwrap();
            }
            1 => {
                let buffer = ImageBuffer::<Luma<u8>, _>::from_raw(
                    self.width,
                    self.height,
                    self.data.clone(),
                )
                .unwrap();
                buffer.save(name).unwrap();
            }
            _ => {
                panic!("Unsupported pixel stride: {}", self.pixel_stride);
            }
        }
    }
}
