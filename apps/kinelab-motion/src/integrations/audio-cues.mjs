/**
 * Lightweight, zero-dependency procedural audio cues and voice coaching.
 * Uses the Web Audio API for instantaneous acoustic biofeedback (depth chime,
 * lockout tick, rep chime) and Web Speech API for hands-free voice coaching.
 */

export class AudioCueSynthesizer {
  constructor({ enabled = true, voiceCoach = false } = {}) {
    this.enabled = enabled;
    this.voiceCoach = voiceCoach;
    this.ctx = null;
    this.lastSpokenText = '';
    this.lastSpokenAt = 0;
  }

  ensureContext() {
    if (!this.enabled || typeof window === 'undefined') return null;
    if (!this.ctx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass) {
        this.ctx = new AudioContextClass();
      }
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume().catch(() => {});
    }
    return this.ctx;
  }

  setEnabled(enabled) {
    this.enabled = Boolean(enabled);
  }

  setVoiceCoachEnabled(enabled) {
    this.voiceCoach = Boolean(enabled);
  }

  /**
   * Ascending two-tone chime for counted repetitions.
   */
  playRepCount() {
    const ctx = this.ensureContext();
    if (!ctx) return;

    try {
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(587.33, now); // D5
      osc.frequency.setValueAtTime(880.0, now + 0.08); // A5

      gain.gain.setValueAtTime(0.001, now);
      gain.gain.exponentialRampToValueAtTime(0.15, now + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.22);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(now);
      osc.stop(now + 0.24);
    } catch (err) {
      console.warn('Audio playback suppressed:', err);
    }
  }

  /**
   * Crystalline high ping (1046.5Hz C6) when reaching valid inflection depth.
   * Gives instant biofeedback without needing to stare at the screen.
   */
  playDepthTarget() {
    const ctx = this.ensureContext();
    if (!ctx) return;

    try {
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(1046.5, now); // C6

      gain.gain.setValueAtTime(0.001, now);
      gain.gain.exponentialRampToValueAtTime(0.18, now + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.12);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(now);
      osc.stop(now + 0.13);
    } catch {}
  }

  /**
   * Crisp lockout tick (523Hz C5) when returning to full extension.
   */
  playLockoutTick() {
    const ctx = this.ensureContext();
    if (!ctx) return;

    try {
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'triangle';
      osc.frequency.setValueAtTime(523.25, now); // C5

      gain.gain.setValueAtTime(0.001, now);
      gain.gain.linearRampToValueAtTime(0.09, now + 0.006);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.06);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(now);
      osc.stop(now + 0.065);
    } catch {}
  }

  /**
   * Soft click for timed hold seconds.
   */
  playHoldTick() {
    const ctx = this.ensureContext();
    if (!ctx) return;

    try {
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'triangle';
      osc.frequency.setValueAtTime(987.77, now); // B5

      gain.gain.setValueAtTime(0.001, now);
      gain.gain.linearRampToValueAtTime(0.06, now + 0.005);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.04);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(now);
      osc.stop(now + 0.045);
    } catch {}
  }

  /**
   * Soft lower warning chime for tracking loss.
   */
  playTrackingLost() {
    const ctx = this.ensureContext();
    if (!ctx) return;

    try {
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(329.63, now); // E4
      osc.frequency.setValueAtTime(261.63, now + 0.08); // C4

      gain.gain.setValueAtTime(0.001, now);
      gain.gain.linearRampToValueAtTime(0.08, now + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.25);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(now);
      osc.stop(now + 0.26);
    } catch {}
  }

  /**
   * Voice Coach: Speaks text using Web Speech API (speechSynthesis).
   */
  speak(text) {
    if (!this.enabled || !this.voiceCoach || typeof window === 'undefined' || !window.speechSynthesis) {
      return;
    }

    const now = performance.now();
    if (this.lastSpokenText === text && now - this.lastSpokenAt < 1200) {
      return; // Debounce rapid repeat speech
    }

    this.lastSpokenText = text;
    this.lastSpokenAt = now;

    try {
      window.speechSynthesis.cancel(); // Cancel any lingering utterance
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.1; // Slightly brisk athletic cadence
      utterance.pitch = 1.0;
      utterance.volume = 0.85;
      window.speechSynthesis.speak(utterance);
    } catch {}
  }

  /**
   * Voice coach rep count announcement.
   */
  speakRep(repCount) {
    if (repCount > 0) {
      this.speak(String(repCount));
    }
  }
}

export const defaultAudioSynthesizer = new AudioCueSynthesizer();
