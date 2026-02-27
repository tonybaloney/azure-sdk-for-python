/** A single chat message exchanged between user and assistant. */
export interface Message {
  id: string
  role: 'user' | 'assistant' | 'error'
  content: string
}
