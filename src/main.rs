use image::{Pixel, Rgba};
use bench_wgpu_buffer::App;

async fn run() {
    let app = App::new(Some(wgpu::Backend::Dx12.into())).await;
    let texture_size = (2048, 2048);
    let (output_buffer, _) = app.create_render_buffer(texture_size);
    // app.render_image(texture_size, output_buffer).await;
    {
        let mask = app
            .generate_fill_mask(
                texture_size,
                (texture_size.0 / 2, texture_size.1 / 2),
                0.01,
                &output_buffer,
            )
            .await;
        mask.save_as_image("mask_direct.png");
    }

    let raw_buffer = app.extract_buffer_data(texture_size, &output_buffer).await;
    {
        let mut raw_buffer = raw_buffer.clone();
        let ts = std::time::Instant::now();
        raw_buffer.flood_fill(
            texture_size.0 / 2,
            texture_size.1 / 2,
            *Rgba::from_slice(&[255, 0, 0, 255]),
            0.01,
        );
        println!("flood fill time: {:?}", ts.elapsed());
        raw_buffer.save_as_image("output.png");
    }
    {
        let ts = std::time::Instant::now();
        let mask = raw_buffer.flood_fill_to_mask(
            texture_size.0 / 2,
            texture_size.1 / 2,
            0.01,
        );
        println!("flood fill to mask time: {:?}", ts.elapsed());
        mask.save_as_image("output_mask.png");
    }
    // scanline flood fill
    {
        let ts = std::time::Instant::now();
        let mask = raw_buffer.flood_fill_to_mask_scanline(texture_size.0 / 2, texture_size.1 / 2, 0.01);
        println!("scanline flood fill time: {:?}", ts.elapsed());
        mask.save_as_image("output_scanline.png");
    }

    // {
    //     let ts = std::time::Instant::now();
    //     let mask = raw_buffer.flood_fill_to_mask(
    //         texture_size.0 / 2,
    //         texture_size.1 / 2,
    //         0.01,
    //     );
    //     println!("flood fill to mask time: {:?}", ts.elapsed());
    //     mask.save_as_image("mask.png");
    // }
}

fn main() {
    pollster::block_on(run());
}
