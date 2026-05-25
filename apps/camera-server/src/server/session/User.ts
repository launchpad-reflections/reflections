import { AppSession } from "@mentra/sdk";
import { PhotoManager } from "../manager/PhotoManager";
import { AudioManager } from "../manager/AudioManager";
import { StorageManager } from "../manager/StorageManager";
import { InputManager } from "../manager/InputManager";

/**
 * User — per-user state container.
 *
 * Composes all managers and holds the glasses AppSession.
 * Created when a user connects and destroyed when the session is cleaned up.
 */
export class User {
  /** Active glasses connection, null when disconnected */
  appSession: AppSession | null = null;

  /** Photo capture and storage */
  photo: PhotoManager;

  /** Text-to-speech and audio control */
  audio: AudioManager;

  /** User preferences via MentraOS Simple Storage */
  storage: StorageManager;

  /** Button presses and touchpad gestures */
  input: InputManager;

  constructor(public readonly userId: string) {
    this.photo = new PhotoManager(this);
    this.audio = new AudioManager(this);
    this.storage = new StorageManager(this);
    this.input = new InputManager(this);
  }

  /** Wire up a glasses connection — sets up all event listeners */
  setAppSession(session: AppSession): void {
    this.appSession = session;
    this.input.setup(session);
    console.log(`📸 Camera ready for ${this.userId}`);
  }

  /** Disconnect glasses but keep user alive */
  clearAppSession(): void {
    this.appSession = null;
  }

  /** Nuke everything — call on full disconnect */
  cleanup(): void {
    this.photo.destroy();
    this.appSession = null;
  }
}
