import { useQuery } from '@tanstack/react-query'
import type { AssetDoc, IndexDoc } from './types'

export function useIndex() {
  return useQuery<IndexDoc>({
    queryKey: ['index'],
    queryFn: async () => {
      const res = await fetch('/index.json', { cache: 'no-store' })
      if (!res.ok) throw new Error(`index.json: HTTP ${res.status}`)
      return res.json()
    },
  })
}

export function useAsset(file: string | null) {
  return useQuery<AssetDoc>({
    queryKey: ['asset', file],
    enabled: file != null,
    queryFn: async () => {
      const res = await fetch(`/${file}`)
      if (!res.ok) throw new Error(`${file}: HTTP ${res.status}`)
      return res.json()
    },
  })
}
