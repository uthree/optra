//! One-way OSC over UDP.
//!
//! Both tracker backends speak OSC to a socket on this machine and neither
//! listens for anything, so the connection is the same object twice: bind a
//! port nobody will ever send to, point it at the consumer, and encode messages
//! into a buffer that gets reused. Only the addresses and the arguments differ,
//! and those are the parts that belong to the backend.
//!
//! It was two copies before, down to the error strings and a comment, and the
//! cost of that is not the duplication. It is that a fix to one of them is not
//! a fix to the other, and the two backends are exactly the pair a user
//! switches between when something looks wrong.

use std::net::{ToSocketAddrs, UdpSocket};

use anyhow::{Context, Result};
use rosc::{OscMessage, OscPacket, OscType, encoder};

/// Room for the longest message either backend sends, which is a twelve-float
/// room matrix. Sized once so that ninety sends a second never reallocate.
const BUFFER: usize = 192;

pub struct OscSender {
    socket: UdpSocket,
    /// What the user typed, kept for reporting rather than for sending.
    target: String,
    buffer: Vec<u8>,
}

impl OscSender {
    /// Points a socket at `target`, which is `host:port`.
    pub fn open(target: &str) -> Result<Self> {
        // Bound to an ephemeral port on the loopback-capable wildcard: nothing
        // is ever received on it, and picking a fixed port would collide with
        // whatever else on the machine is speaking OSC.
        let socket = UdpSocket::bind("0.0.0.0:0").context("could not open a UDP socket")?;

        let resolved = target
            .to_socket_addrs()
            .with_context(|| format!("{target} is not an address"))?
            .next()
            .with_context(|| format!("{target} resolved to nothing"))?;
        socket
            .connect(resolved)
            .with_context(|| format!("could not point a socket at {target}"))?;

        Ok(Self {
            socket,
            target: target.to_owned(),
            buffer: Vec::with_capacity(BUFFER),
        })
    }

    pub fn target(&self) -> &str {
        &self.target
    }

    pub fn send(&mut self, address: &str, args: Vec<OscType>) -> Result<()> {
        let packet = OscPacket::Message(OscMessage {
            addr: address.to_owned(),
            args,
        });

        self.buffer.clear();
        // Encoding into a `Vec` cannot fail; the error type is `Infallible`.
        encoder::encode_into(&packet, &mut self.buffer).ok();
        self.socket
            .send(&self.buffer)
            .with_context(|| format!("could not send {address}"))?;
        Ok(())
    }

    /// Three floats, which is how both backends write a position or an angle.
    pub fn send_triple(&mut self, address: &str, x: f64, y: f64, z: f64) -> Result<()> {
        self.send(
            address,
            vec![
                OscType::Float(x as f32),
                OscType::Float(y as f32),
                OscType::Float(z as f32),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Binds a receiver and returns it with the address to aim at it.
    fn listener() -> (UdpSocket, String) {
        let socket = UdpSocket::bind("127.0.0.1:0").expect("a loopback port");
        let address = socket.local_addr().expect("its own address").to_string();
        (socket, address)
    }

    fn received(socket: &UdpSocket) -> OscMessage {
        let mut bytes = [0u8; 1024];
        let (length, _) = socket.recv_from(&mut bytes).expect("a packet");
        match rosc::decoder::decode_udp(&bytes[..length]) {
            Ok((_, OscPacket::Message(message))) => message,
            other => panic!("expected one message, got {other:?}"),
        }
    }

    #[test]
    fn a_message_arrives_with_its_address_and_arguments() {
        let (socket, address) = listener();
        let mut sender = OscSender::open(&address).expect("the sender should open");
        assert_eq!(sender.target(), address);

        sender
            .send("/test/one", vec![OscType::Int(7), OscType::Float(1.5)])
            .expect("the send should succeed");

        let message = received(&socket);
        assert_eq!(message.addr, "/test/one");
        assert_eq!(message.args, vec![OscType::Int(7), OscType::Float(1.5)]);
    }

    #[test]
    fn a_triple_arrives_as_three_floats() {
        let (socket, address) = listener();
        let mut sender = OscSender::open(&address).expect("the sender should open");

        sender
            .send_triple("/test/triple", 1.0, -2.0, 3.5)
            .expect("the send should succeed");

        let message = received(&socket);
        assert_eq!(
            message.args,
            vec![
                OscType::Float(1.0),
                OscType::Float(-2.0),
                OscType::Float(3.5)
            ]
        );
    }

    /// The buffer is reused, so a short message after a long one must not
    /// carry the tail of the long one with it.
    #[test]
    fn a_short_message_after_a_long_one_carries_nothing_of_it() {
        let (socket, address) = listener();
        let mut sender = OscSender::open(&address).expect("the sender should open");

        sender
            .send(
                "/test/long",
                (0..12).map(|i| OscType::Float(i as f32)).collect(),
            )
            .expect("the send should succeed");
        received(&socket);

        sender
            .send("/test/short", vec![OscType::Int(1)])
            .expect("the send should succeed");

        let message = received(&socket);
        assert_eq!(message.addr, "/test/short");
        assert_eq!(message.args, vec![OscType::Int(1)]);
    }

    #[test]
    fn an_address_that_resolves_to_nothing_is_refused() {
        assert!(OscSender::open("not a host name at all").is_err());
    }
}
