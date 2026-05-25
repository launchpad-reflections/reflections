import type { User } from "../session/User";

export interface StoredPhoto {
  requestId: string;
  buffer: Buffer;
  timestamp: Date;
  userId: string;
  mimeType: string;
  filename: string;
  size: number;
}

/**
 * PhotoManager — captures photos from the glasses (button/tap triggered
 * via InputManager) and caches them in memory for the active session.
 *
 * Photos are kept in-process so a downstream consumer (e.g. the Python
 * viewer fetching a snapshot, or a future REST endpoint) can pick them
 * up without coupling capture to network delivery.
 */
export class PhotoManager {
  private photos: Map<string, StoredPhoto> = new Map();

  constructor(private user: User) {}

  /** Capture a photo from the glasses and store it. */
  async takePhoto(): Promise<void> {
    const session = this.user.appSession;
    if (!session) throw new Error("No active glasses session");

    const photo = await session.camera.requestPhoto();

    const stored: StoredPhoto = {
      requestId: photo.requestId,
      buffer: photo.buffer,
      timestamp: photo.timestamp,
      userId: this.user.userId,
      mimeType: photo.mimeType,
      filename: photo.filename,
      size: photo.size,
    };

    this.photos.set(photo.requestId, stored);
    console.log(
      `📸 Photo captured for ${this.user.userId} (${photo.size} bytes)`,
    );
  }

  getPhoto(requestId: string): StoredPhoto | undefined {
    return this.photos.get(requestId);
  }

  /** All photos for this user, sorted newest-first. */
  getAll(): StoredPhoto[] {
    return Array.from(this.photos.values()).sort(
      (a, b) => b.timestamp.getTime() - a.timestamp.getTime(),
    );
  }

  removeAll(): void {
    this.photos.clear();
  }

  /** Tear down — clear cached photos. */
  destroy(): void {
    this.photos.clear();
  }
}
