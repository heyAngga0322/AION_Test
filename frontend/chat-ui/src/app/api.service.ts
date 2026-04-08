import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface QueryResponse {
  answer: string;
  sources: string[];
}

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private http = inject(HttpClient);
  private baseUrl = '/api';

  uploadDocument(title: string, file: File | null, textContent: string | null): Observable<any> {
    const formData = new FormData();
    formData.append('title', title);
    if (file) {
      formData.append('file', file);
    }
    if (textContent) {
      formData.append('text_content', textContent);
    }
    return this.http.post(`${this.baseUrl}/documents`, formData);
  }

  askQuestion(question: string): Observable<QueryResponse> {
    return this.http.post<QueryResponse>(`${this.baseUrl}/ask`, { question });
  }
}
