// AudioWorklet (runs on the browser's audio thread): batches raw samples of one stream and posts them to the page.
// The page does voice detection and WAV encoding; this file only moves samples (D32).
class FrameTap extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(2048);  // ~43 ms at 48 kHz: small enough for snappy detection, few messages
    this.n = 0;
  }

  process(inputs) {
    const channels = inputs[0];
    if (channels && channels.length) {
      const frame = channels[0];
      for (let i = 0; i < frame.length; i++) {
        // mono mix: average all channels (tab audio is often stereo)
        let s = 0;
        for (let c = 0; c < channels.length; c++) s += channels[c][i];
        this.buf[this.n++] = s / channels.length;
        if (this.n === this.buf.length) {
          this.port.postMessage(this.buf.slice(0));
          this.n = 0;
        }
      }
    }
    return true;  // keep the processor alive while the stream runs
  }
}

registerProcessor("frame-tap", FrameTap);
