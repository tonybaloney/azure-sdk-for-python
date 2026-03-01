import { useState, useRef, useCallback } from 'react'
import type { Message } from '../types'

/**
 * Hook managing chat messages via the Responses API with SSE streaming.
 *
 * The agent exposes POST /responses which returns SSE events in the
 * OpenAI Responses API format.  Text deltas arrive as events with
 * type "response.output_text.delta" and a "delta" field.
 */
export function useChat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const messagesRef = useRef<Message[]>([])
  const abortRef = useRef<AbortController | null>(null)

  const sendMessage = useCallback(async (text: string) => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    const userMsg: Message = { id: crypto.randomUUID(), role: 'user', content: text }
    const assistantId = crypto.randomUUID()
    const assistantMsg: Message = { id: assistantId, role: 'assistant', content: '' }

    messagesRef.current = [...messagesRef.current, userMsg, assistantMsg]
    setMessages([...messagesRef.current])
    setIsLoading(true)

    try {
      const res = await fetch('/responses', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: text, stream: true }),
        signal: controller.signal,
      })

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`)
      }

      const reader = res.body?.getReader()
      const decoder = new TextDecoder()
      let content = ''
      let buffer = ''

      if (reader) {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() ?? ''

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            const data = line.slice(6)
            if (data === '[DONE]') continue

            try {
              const event = JSON.parse(data)

              // Extract text from delta events
              if (event.type === 'response.output_text.delta' && event.delta) {
                content += event.delta
                messagesRef.current = messagesRef.current.map(m =>
                  m.id === assistantId ? { ...m, content } : m,
                )
                setMessages([...messagesRef.current])
              }

              // Also handle completed output items with full text
              if (event.type === 'response.output_item.done' && event.item?.content) {
                for (const part of event.item.content) {
                  if (part.type === 'output_text' && part.text && !content) {
                    content = part.text
                    messagesRef.current = messagesRef.current.map(m =>
                      m.id === assistantId ? { ...m, content } : m,
                    )
                    setMessages([...messagesRef.current])
                  }
                }
              }
            } catch (e) {
              if (e instanceof SyntaxError) continue
              throw e
            }
          }
        }
      }

      if (!content) {
        // Try non-streaming: the response may be a direct JSON object
        messagesRef.current = messagesRef.current.map(m =>
          m.id === assistantId ? { ...m, content: '(empty response)' } : m,
        )
        setMessages([...messagesRef.current])
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return
      messagesRef.current = messagesRef.current.map(m =>
        m.id === assistantId
          ? { ...m, role: 'error' as const, content: err instanceof Error ? err.message : 'Unknown error' }
          : m,
      )
      setMessages([...messagesRef.current])
    } finally {
      setIsLoading(false)
    }
  }, [])

  return { messages, isLoading, sendMessage }
}
