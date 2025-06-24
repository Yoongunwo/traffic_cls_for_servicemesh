// Full working Rust proxy with inference
// Cargo.toml dependencies:
// [dependencies]
// tokio = { version = "1.37", features = ["full"] }
// tch = { git = "https://github.com/LaurentMazare/tch-rs.git" }
// torch-sys = { git = "https://github.com/LaurentMazare/tch-rs.git", package = "torch-sys" }
// anyhow = "1.0"

use anyhow::Result;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tch::{CModule, Kind, Tensor};

const VEC_LEN: usize = 1479;
const WIN_SIZE: usize = 15;
const H: usize = 34;
const W: usize = 44;
const MAX_SESSIONS: usize = 65536;

type SessionBuffer = [[f32; VEC_LEN]; WIN_SIZE];

fn extract_session_id(packet: &[u8]) -> usize {
    if packet.len() < 38 {
        return 0;
    }
    let src_ip = u32::from_be_bytes([packet[26], packet[27], packet[28], packet[29]]);
    let dst_ip = u32::from_be_bytes([packet[30], packet[31], packet[32], packet[33]]);
    let src_port = u16::from_be_bytes([packet[34], packet[35]]) as u32;
    let dst_port = u16::from_be_bytes([packet[36], packet[37]]) as u32;
    let proto = packet[23] as u32;
    ((src_ip ^ dst_ip ^ src_port ^ dst_port ^ proto) as usize) % MAX_SESSIONS
}

fn parse_payload(data: &[u8]) -> Option<[f32; VEC_LEN]> {
    if data.is_empty() {
        return None;
    }
    let mut out = [0f32; VEC_LEN];
    for (i, &b) in data.iter().take(VEC_LEN).enumerate() {
        out[i] = b as f32;
    }
    Some(out)
}

async fn handle_connection(
    mut client: TcpStream,
    model: Arc<CModule>,
    target: SocketAddr,
    session_buf: Arc<Mutex<Vec<SessionBuffer>>>,
    session_counts: Arc<Mutex<[usize; MAX_SESSIONS]>>,
) -> Result<()> {
    let mut server = TcpStream::connect(target).await?;
    let (mut r1, mut w1) = client.split();
    let (mut r2, mut w2) = server.split();
    let model_clone = model.clone();
    let buf_clone = session_buf.clone();
    let count_clone = session_counts.clone();

    let c2s = async move {
        let mut buf = [0u8; 4096];
        while let Ok(n) = r1.read(&mut buf).await {
            if n == 0 {
                break;
            }
            let packet = &buf[..n];

            if let Some(vec) = parse_payload(packet) {
                let session_id = extract_session_id(packet);

                {
                    let mut buffers = buf_clone.lock().unwrap();
                    let mut counts = count_clone.lock().unwrap();
                    let idx = counts[session_id];

                    if idx >= WIN_SIZE {
                        buffers[session_id].copy_within(1..WIN_SIZE, 0);
                        buffers[session_id][WIN_SIZE - 1] = vec;
                    } else {
                        buffers[session_id][idx] = vec;
                        counts[session_id] += 1;
                    }

                    if counts[session_id] == WIN_SIZE {
                        let flat: Vec<f32> = buffers[session_id].iter().flatten().copied().collect();
                        drop(buffers);
                        drop(counts);

                        let input = Tensor::from_slice(&flat[..H * W])
                            .view([1, 1, H as i64, W as i64]) / 255.0;

                        let logits = model_clone.forward_ts(&[input])?;
                        let probs = logits.softmax(-1, Kind::Float);

                        // (값, 인덱스) 튜플
                        let (top_prob, top_idx) = probs.max_dim(-1, false);

                        // ✅ 값 추출 (index 0 위치)
                        let prob_value = top_prob.double_value(&[0]) as f32;
                        let idx_value = top_idx.int64_value(&[0]);

                        println!("[Model Result] class = {}, prob = {:.4}", idx_value, prob_value);
                    }
                }

                w2.write_all(packet).await?;
            }
        }
        Ok::<_, anyhow::Error>(())
    };

    let s2c = async move {
        let mut buf = [0u8; 4096];
        while let Ok(n) = r2.read(&mut buf).await {
            if n == 0 {
                break;
            }
            w1.write_all(&buf[..n]).await?;
        }
        Ok::<_, anyhow::Error>(())
    };

    tokio::try_join!(c2s, s2c)?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = Arc::new(CModule::load("/app/Model/student_encoder_ts.pt")?);
    let session_buf = Arc::new(Mutex::new(vec![[[0.0; VEC_LEN]; WIN_SIZE]; MAX_SESSIONS]));
    let session_counts = Arc::new(Mutex::new([0; MAX_SESSIONS]));

    let listener = TcpListener::bind("0.0.0.0:8080").await?;
    println!("Proxy listening on 0.0.0.0:8080");
    let target: SocketAddr = "127.0.0.1:80".parse()?;

    loop {
        let (client, _) = listener.accept().await?;
        let model = model.clone();
        let buf = session_buf.clone();
        let counts = session_counts.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(client, model, target, buf, counts).await {
                eprintln!("Connection error: {}", e);
            }
        });
    }
}