import { Component, signal, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService, QueryResponse } from '../api.service';

@Component({
  selector: 'app-chat',
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.html',
  styleUrl: './chat.css',
})
export class Chat {
  apiService = inject(ApiService);

  // Upload State
  uploadTitle = signal('');
  uploadText = signal('');
  selectedFile = signal<File | null>(null);
  uploadStatus = signal('');
  isUploading = signal(false);

  // Chat State
  question = signal('');
  chatHistory = signal<{ role: 'user' | 'system', content: string, sources?: string[] }[]>([]);
  isAsking = signal(false);

  onFileSelected(event: any) {
    const file = event.target.files[0];
    if (file) {
      this.selectedFile.set(file);
    }
  }

  async onUpload() {
    if (!this.uploadTitle()) {
      this.uploadStatus.set('Please provide a title.');
      return;
    }
    if (!this.selectedFile() && !this.uploadText()) {
      this.uploadStatus.set('Please provide a file or text content.');
      return;
    }

    this.isUploading.set(true);
    this.uploadStatus.set('Uploading...');

    this.apiService.uploadDocument(this.uploadTitle(), this.selectedFile(), this.uploadText())
      .subscribe({
        next: (res) => {
          this.uploadStatus.set('Successfully uploaded and processed.');
          this.isUploading.set(false);
          this.uploadTitle.set('');
          this.uploadText.set('');
          this.selectedFile.set(null);
        },
        error: (err) => {
          this.uploadStatus.set('Error uploading document.');
          this.isUploading.set(false);
        }
      });
  }

  async onAsk() {
    const q = this.question().trim();
    if (!q) return;

    this.chatHistory.update(h => [...h, { role: 'user', content: q }]);
    this.question.set('');
    this.isAsking.set(true);

    this.apiService.askQuestion(q).subscribe({
      next: (res: QueryResponse) => {
        this.chatHistory.update(h => [...h, { role: 'system', content: res.answer, sources: res.sources }]);
        this.isAsking.set(false);
      },
      error: (err) => {
        this.chatHistory.update(h => [...h, { role: 'system', content: 'Sorry, there was an error processing your question. Ensure the backend is running.' }]);
        this.isAsking.set(false);
      }
    });
  }
}
