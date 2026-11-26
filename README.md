# Professor de Xadrez

Revisão de partidas no estilo chess.com, com professor por voz.

- Cole o link da partida do chess.com ou o PGN e clique em **Analisar**.
- Stockfish 18/19 roda no navegador (vários núcleos quando o computador tem 8 ou mais); voz Dora (Kokoro-82M) também roda no navegador.
- **♟ Jogar**: partida contra o computador — adversários humanos (Maia 3, rating 600–2600) ou Stockfish (Elo 1320–3190 ou força máxima), relógio, cor, posição inicial ou FEN, ajudas; no fim, "Analisar partida" leva para a revisão.
- Professor Claude: cole a sua chave da API da Anthropic no ⚙ (fica só no seu navegador).

Créditos: Stockfish (GPL-3.0), chessground / lichess (GPL-3.0), chess.js (BSD-2), onnxruntime-web (MIT), Kokoro-82M (Apache-2.0), piper-phonemize / espeak-ng (GPL-3.0), adversário humano Maia 3 (CSSLab/Universidade de Toronto, `maia3.onnx` do maia-platform-frontend, GPL-3.0), sons das peças do lichess `public/sound/standard` (AGPL-3.0+), coi-serviceworker (MIT, adaptado em `coi-sw.js`).
