export type WebRTCState = 'connecting' | 'signaling' | 'receiving' | 'playing' | 'failed' | 'disconnected'

export class WebRTCManager {
  private socket: WebSocket | null = null
  private peer: RTCPeerConnection | null = null
  private pendingStart = false

  constructor(
    private readonly onStream: (stream: MediaStream) => void,
    private readonly onState: (state: WebRTCState) => void,
  ) {}

  connect(): void {
    this.onState('connecting')
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    this.socket = new WebSocket(`${protocol}//${window.location.host}/ws/webrtc`)
    this.socket.onopen = () => {
      this.onState('signaling')
      if (this.pendingStart) this.startCombinedStream().catch(console.error)
    }
    this.socket.onmessage = event => {
      const message = JSON.parse(event.data) as { type: string; sdp?: string }
      if (message.type === 'answer' && message.sdp && this.peer) {
        this.peer.setRemoteDescription({ type: 'answer', sdp: message.sdp }).catch(error => {
          this.onState('failed')
          console.error(error)
        })
      }
      if (message.type === 'error') this.onState('failed')
      if (message.type === 'ping') this.socket?.send(JSON.stringify({ type: 'pong' }))
    }
    this.socket.onerror = () => this.onState('failed')
    this.socket.onclose = () => this.onState('disconnected')
  }

  async startCombinedStream(): Promise<void> {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      this.pendingStart = true
      return
    }
    this.pendingStart = false
    this.peer?.close()
    const peer = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] })
    this.peer = peer
    peer.onconnectionstatechange = () => {
      if (peer.connectionState === 'failed') this.onState('failed')
      if (peer.connectionState === 'disconnected' || peer.connectionState === 'closed') {
        this.onState('disconnected')
      }
    }
    peer.addTransceiver('video', { direction: 'recvonly' })
    peer.ontrack = event => {
      const stream = event.streams[0] ?? new MediaStream([event.track])
      this.onStream(stream)
      this.onState('receiving')
    }
    const offer = await peer.createOffer()
    await peer.setLocalDescription(offer)
    await new Promise<void>(resolve => {
      if (peer.iceGatheringState === 'complete') return resolve()
      const timeout = window.setTimeout(resolve, 2000)
      peer.addEventListener('icegatheringstatechange', () => {
        if (peer.iceGatheringState === 'complete') { window.clearTimeout(timeout); resolve() }
      })
    })
    this.socket.send(JSON.stringify({ type: 'offer', sdp: peer.localDescription?.sdp }))
  }

  disconnect(): void {
    this.pendingStart = false
    this.peer?.close()
    this.peer = null
    this.socket?.close(1000)
    this.socket = null
    this.onState('disconnected')
  }
}
