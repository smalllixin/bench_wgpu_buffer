use criterion::BenchmarkId;
use criterion::{criterion_group, criterion_main, Criterion};
// This is a struct that tells Criterion.rs to use the "futures" crate's current-thread executor
use criterion::async_executor::FuturesExecutor;
use criterion::Throughput;
use bench_wgpu_buffer::{App, RawBuffer};

fn mapping_gpubuffer_gl(c: &mut Criterion) {
    let app = pollster::block_on(App::new(Some(wgpu::Backend::Gl.into())));
    let texture_sizes = vec![(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)];

    let mut group = c.benchmark_group("mapping_buffer_gl");

    for texture_size in texture_sizes {
        let bench_name = format!(
            "{}_mapping_buffer_{}x{}",
            app.to_string(),
            texture_size.0,
            texture_size.1
        );
        group.throughput(Throughput::Bytes(
            texture_size.0 as u64 * texture_size.1 as u64 * 4,
        ));
        group.bench_with_input(
            BenchmarkId::new("mapping_buffer", bench_name),
            &app,
            |b, app| {
                b.to_async(FuturesExecutor).iter_batched(
                    || app.create_render_buffer(texture_size),
                    |(output_buffer, _)| async move {
                        let d = app.just_map_buffer(output_buffer).await;
                        let _ = d[0];
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn mapping_gpubuffer_dx12(c: &mut Criterion) {
    let app = pollster::block_on(App::new(Some(wgpu::Backend::Dx12.into())));
    let texture_sizes = vec![(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)];

    let mut group = c.benchmark_group("mapping_buffer_dx12");

    for texture_size in texture_sizes {
        let bench_name = format!(
            "{}_mapping_buffer_{}x{}",
            app.to_string(),
            texture_size.0,
            texture_size.1
        );
        group.throughput(Throughput::Bytes(
            texture_size.0 as u64 * texture_size.1 as u64 * 4,
        ));
        group.bench_with_input(
            BenchmarkId::new("mapping_buffer", bench_name),
            &app,
            |b, app| {
                b.to_async(FuturesExecutor).iter_batched(
                    || app.create_render_buffer(texture_size),
                    |(output_buffer, _)| async move { app.just_map_buffer(output_buffer).await },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

fn flood_fill_benchmark(c: &mut Criterion) {
    let texture_sizes = vec![(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)];

    let mut raw_buffers = Vec::new();

    for texture_size in &texture_sizes {
        let mut raw_buffer = RawBuffer::new(
            vec![0; texture_size.0 as usize * texture_size.1 as usize * 4],
            texture_size.0,
            texture_size.1,
            4,
        );

        // Fill the buffer with yellow color
        for pixel in raw_buffer.data.chunks_exact_mut(4) {
            pixel.copy_from_slice(&[255, 255, 0, 255]); // Yellow color (RGBA)
        }

        // Draw a white circle
        let center_x = texture_size.0 / 2;
        let center_y = texture_size.1 / 2;
        let radius = texture_size.0.min(texture_size.1) / 2;

        for y in 0..texture_size.1 {
            for x in 0..texture_size.0 {
                let dx = x as i32 - center_x as i32;
                let dy = y as i32 - center_y as i32;
                if dx * dx + dy * dy <= (radius as i32).pow(2) {
                    let index = (y * texture_size.0 + x) as usize * 4;
                    raw_buffer.data[index..index + 4].copy_from_slice(&[255, 255, 255, 255]);
                    // White color (RGBA)
                }
            }
        }

        raw_buffers.push(raw_buffer);
    }

    let mut group = c.benchmark_group("flood_fill");
    for (i, texture_size) in texture_sizes.iter().enumerate() {
        let raw_buffer = &raw_buffers[i];
        group.bench_function(
            &format!(
                "generate_fill_mask_raw_{}x{}",
                texture_size.0, texture_size.1
            ),
            |b| {
                b.iter(|| {
                    raw_buffer.flood_fill_to_mask(texture_size.0 / 2, texture_size.1 / 2, 0.01)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, mapping_gpubuffer_gl, mapping_gpubuffer_dx12);
criterion_group!(flood_fill, flood_fill_benchmark);
criterion_main!(benches, flood_fill);
