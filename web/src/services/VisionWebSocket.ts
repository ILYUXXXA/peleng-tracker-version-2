export interface StreamTrack { id: string; bbox: [number, number, number, number]; confidence: number; class_name: string; class_id: number; center: [number, number]; global_id: string | null; is_target: boolean }
export interface StreamData { detections: StreamTrack[]; tracks: StreamTrack[] }
export interface WorldTrack {
  global_id: string; class_id: number; class_name: string
  position_xyz_m: [number, number, number] | null
  velocity_xyz_mps: [number, number, number]; confidence: number
  source_cameras: number[]; reprojection_error_px: number | null
  position_status: 'valid' | 'planar_fallback' | 'unavailable'
  is_target: boolean; trajectory: Array<[number, number, number, number]>
  source_track_keys: string[]
  association_pairs: Array<{
    left_camera_id: number; left_local_track_id: string
    right_camera_id: number; right_local_track_id: string
    ground_distance_m: number | null; reprojection_error_px: number; cost: number
  }>
}
export interface VisionMetadata {
  type: 'vision_metadata'; schema_version: 2; timestamp: number; frame_id: number
  processing_time_ms: number; sync_skew_ms: number; active_stream_ids: number[]
  association_diagnostics: { local_tracks_by_camera: Record<string, number>; fused_world_tracks: number; single_camera_fallback_tracks: number }
  all_streams: Record<string, StreamData>; world_tracks: WorldTrack[]; bev_tracks: WorldTrack[]
}
export interface VisionStatus { type: 'vision_status'; schema_version: 2; timestamp: number; tracking_enabled: boolean; active_stream_ids: number[]; message: string }
export type VisionMessage = VisionMetadata | VisionStatus

export class VisionWebSocketService {
  private websocket: WebSocket | null = null
  private retryTimer: number | null = null
  private closedByClient = false
  public onMessage?: (data: VisionMessage) => void
  public onConnect?: () => void
  public onDisconnect?: () => void

  connect(): void {
    if (this.websocket && this.websocket.readyState < WebSocket.CLOSING) return
    this.closedByClient = false
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    this.websocket = new WebSocket(`${protocol}//${window.location.host}/ws/vision-metadata`)
    this.websocket.onopen = () => this.onConnect?.()
    this.websocket.onmessage = event => {
      try { this.onMessage?.(JSON.parse(event.data) as VisionMessage) }
      catch (error) { console.error('Invalid vision payload', error) }
    }
    this.websocket.onclose = () => {
      this.websocket = null
      this.onDisconnect?.()
      if (!this.closedByClient) this.retryTimer = window.setTimeout(() => this.connect(), 3000)
    }
  }

  disconnect(): void {
    this.closedByClient = true
    if (this.retryTimer !== null) window.clearTimeout(this.retryTimer)
    this.websocket?.close()
    this.websocket = null
  }
}
