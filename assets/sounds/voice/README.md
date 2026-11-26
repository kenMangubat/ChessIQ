# ChessIQ Voice Clips

Per-bot voice-over WAV clips (`22050 Hz`, mono, 16-bit PCM) loaded by `ui/sound_manager.py`.

## Generation & Provenance
- Generated at build time using `others/generate_voice_clips.py --backend omnivoice-design` via `k2-fsa/OmniVoice` synthetic Filipino AI voice design (`male, middle-aged, low pitch`, seed `77`, native `ceb` / `fil` / `en` synthesis), with subtle per-bot vocal shading (`focus_ken` master, `sick_ken` slower/warmer, `condition_ken` deeper) and `-1.0 dBFS` normalization via `others/clip_audio.py`.
- Direct blunder clip (`walang_kanin_buseng.wav`): sliced from `walang kanen buseng.mp3` (`3.05s–5.32s`) using `others/clip_audio.py` and normalized to `-1.0 dBFS`.
- Audition any clip with: `.\.venv\Scripts\python.exe others/play_clip.py <path-to-wav>`

| File | Bot | Event | Lang | Duration | Date | Text |
|---|---|---|---|---:|---|---|
| `walang_kanin_buseng.wav` | `all` | `PLAYER_BLUNDER` | `fil` | 2.27s | 2026-11-26 | Walang kanin Buseng! |
| `sick_ken/greeting.wav` | `sick_ken` | `GREETING` | `ceb` | 2.37s | 2026-11-26 | Bai gihilantan ko, pildiha daw ko bi |
| `focus_ken/greeting.wav` | `focus_ken` | `GREETING` | `ceb` | 1.98s | 2026-11-26 | Ha?? sure ka makig duwa ka nako |
| `condition_ken/greeting.wav` | `condition_ken` | `GREETING` | `fil` | 1.76s | 2026-11-26 | hmmm dependi kong 3 yan! |
| `sick_ken/blunder_bossing.wav` | `sick_ken` | `PLAYER_BLUNDER` | `fil` | 2.39s | 2026-11-26 | Oyy bossing kamusta kabuhay buhay |
| `focus_ken/blunder_bossing.wav` | `focus_ken` | `PLAYER_BLUNDER` | `fil` | 2.25s | 2026-11-26 | Oyy bossing kamusta kabuhay buhay |
| `condition_ken/blunder_bossing.wav` | `condition_ken` | `PLAYER_BLUNDER` | `fil` | 2.29s | 2026-11-26 | Oyy bossing kamusta kabuhay buhay |
| `sick_ken/blunder_esmeringhoy.wav` | `sick_ken` | `PLAYER_BLUNDER` | `ceb` | 2.96s | 2026-11-26 | Nahhh ma esmeringhoy na imong position ron |
| `focus_ken/blunder_esmeringhoy.wav` | `focus_ken` | `PLAYER_BLUNDER` | `ceb` | 2.76s | 2026-11-26 | Nahhh ma esmeringhoy na imong position ron |
| `condition_ken/blunder_esmeringhoy.wav` | `condition_ken` | `PLAYER_BLUNDER` | `ceb` | 2.81s | 2026-11-26 | Nahhh ma esmeringhoy na imong position ron |
| `sick_ken/mistake_sayop.wav` | `sick_ken` | `PLAYER_MISTAKE` | `ceb` | 1.63s | 2026-11-26 | Sayop man ka buseng! |
| `focus_ken/mistake_sayop.wav` | `focus_ken` | `PLAYER_MISTAKE` | `ceb` | 1.53s | 2026-11-26 | Sayop man ka buseng! |
| `condition_ken/mistake_sayop.wav` | `condition_ken` | `PLAYER_MISTAKE` | `ceb` | 1.57s | 2026-11-26 | Sayop man ka buseng! |
| `sick_ken/mistake_pataka.wav` | `sick_ken` | `PLAYER_MISTAKE` | `ceb` | 1.62s | 2026-11-26 | Nag pataka raman kag tira oy |
| `focus_ken/mistake_pataka.wav` | `focus_ken` | `PLAYER_MISTAKE` | `ceb` | 1.51s | 2026-11-26 | Nag pataka raman kag tira oy |
| `condition_ken/mistake_pataka.wav` | `condition_ken` | `PLAYER_MISTAKE` | `ceb` | 1.54s | 2026-11-26 | Nag pataka raman kag tira oy |
| `sick_ken/mistake_bushet.wav` | `sick_ken` | `PLAYER_MISTAKE` | `ceb` | 1.55s | 2026-11-26 | Bushet manang tiraha oy |
| `focus_ken/mistake_bushet.wav` | `focus_ken` | `PLAYER_MISTAKE` | `ceb` | 1.46s | 2026-11-26 | Bushet manang tiraha oy |
| `condition_ken/mistake_bushet.wav` | `condition_ken` | `PLAYER_MISTAKE` | `ceb` | 1.49s | 2026-11-26 | Bushet manang tiraha oy |
| `sick_ken/dubious_sure_ka.wav` | `sick_ken` | `PLAYER_DUBIOUS` | `ceb` | 1.09s | 2026-11-26 | Ha? sure ka ana |
| `focus_ken/dubious_sure_ka.wav` | `focus_ken` | `PLAYER_DUBIOUS` | `ceb` | 1.03s | 2026-11-26 | Ha? sure ka ana |
| `condition_ken/dubious_sure_ka.wav` | `condition_ken` | `PLAYER_DUBIOUS` | `ceb` | 1.04s | 2026-11-26 | Ha? sure ka ana |
| `sick_ken/dubious_wa_ka.wav` | `sick_ken` | `PLAYER_DUBIOUS` | `ceb` | 1.19s | 2026-11-26 | Wa ka ma bo... |
| `focus_ken/dubious_wa_ka.wav` | `focus_ken` | `PLAYER_DUBIOUS` | `ceb` | 1.11s | 2026-11-26 | Wa ka ma bo... |
| `condition_ken/dubious_wa_ka.wav` | `condition_ken` | `PLAYER_DUBIOUS` | `ceb` | 1.13s | 2026-11-26 | Wa ka ma bo... |
| `sick_ken/dubious_mura_sakto.wav` | `sick_ken` | `PLAYER_DUBIOUS` | `ceb` | 1.41s | 2026-11-26 | Ahhh mura mag sakto |
| `focus_ken/dubious_mura_sakto.wav` | `focus_ken` | `PLAYER_DUBIOUS` | `ceb` | 1.32s | 2026-11-26 | Ahhh mura mag sakto |
| `condition_ken/dubious_mura_sakto.wav` | `condition_ken` | `PLAYER_DUBIOUS` | `ceb` | 1.34s | 2026-11-26 | Ahhh mura mag sakto |
| `sick_ken/great_pa_kiss.wav` | `sick_ken` | `PLAYER_GREAT` | `ceb` | 1.74s | 2026-11-26 | Woww pa kiss ko bi |
| `focus_ken/great_pa_kiss.wav` | `focus_ken` | `PLAYER_GREAT` | `ceb` | 1.62s | 2026-11-26 | Woww pa kiss ko bi |
| `condition_ken/great_pa_kiss.wav` | `condition_ken` | `PLAYER_GREAT` | `ceb` | 1.65s | 2026-11-26 | Woww pa kiss ko bi |
| `sick_ken/great_grabe_sir.wav` | `sick_ken` | `PLAYER_GREAT` | `ceb` | 2.01s | 2026-11-26 | Grabe sirr ayaw ana sir |
| `focus_ken/great_grabe_sir.wav` | `focus_ken` | `PLAYER_GREAT` | `ceb` | 1.89s | 2026-11-26 | Grabe sirr ayaw ana sir |
| `condition_ken/great_grabe_sir.wav` | `condition_ken` | `PLAYER_GREAT` | `ceb` | 1.92s | 2026-11-26 | Grabe sirr ayaw ana sir |
| `sick_ken/brilliant_omg.wav` | `sick_ken` | `PLAYER_BRILLIANT` | `ceb` | 1.89s | 2026-11-26 | OMG! unsa mana tiraha oy |
| `focus_ken/brilliant_omg.wav` | `focus_ken` | `PLAYER_BRILLIANT` | `ceb` | 1.77s | 2026-11-26 | OMG! unsa mana tiraha oy |
| `condition_ken/brilliant_omg.wav` | `condition_ken` | `PLAYER_BRILLIANT` | `ceb` | 1.80s | 2026-11-26 | OMG! unsa mana tiraha oy |
| `sick_ken/brilliant_cheer.wav` | `sick_ken` | `PLAYER_BRILLIANT` | `en` | 1.23s | 2026-11-26 | Briliianttttt |
| `focus_ken/brilliant_cheer.wav` | `focus_ken` | `PLAYER_BRILLIANT` | `en` | 1.14s | 2026-11-26 | Briliianttttt |
| `condition_ken/brilliant_cheer.wav` | `condition_ken` | `PLAYER_BRILLIANT` | `en` | 1.17s | 2026-11-26 | Briliianttttt |
| `sick_ken/brilliant_nakita.wav` | `sick_ken` | `PLAYER_BRILLIANT` | `fil` | 2.05s | 2026-11-26 | Ahh shet shet nakita ako |
| `focus_ken/brilliant_nakita.wav` | `focus_ken` | `PLAYER_BRILLIANT` | `fil` | 1.92s | 2026-11-26 | Ahh shet shet nakita ako |
| `condition_ken/brilliant_nakita.wav` | `condition_ken` | `PLAYER_BRILLIANT` | `fil` | 1.95s | 2026-11-26 | Ahh shet shet nakita ako |
| `sick_ken/check.wav` | `sick_ken` | `CHECK` | `en` | 1.82s | 2026-11-26 | Check! Watch the king. |
| `focus_ken/check.wav` | `focus_ken` | `CHECK` | `en` | 1.70s | 2026-11-26 | Check! Watch the king. |
| `condition_ken/check.wav` | `condition_ken` | `CHECK` | `en` | 1.74s | 2026-11-26 | Check! Watch the king. |
| `sick_ken/promotion.wav` | `sick_ken` | `PROMOTION` | `en` | 1.99s | 2026-11-26 | Fresh queen on the board! |
| `focus_ken/promotion.wav` | `focus_ken` | `PROMOTION` | `en` | 1.86s | 2026-11-26 | Fresh queen on the board! |
| `condition_ken/promotion.wav` | `condition_ken` | `PROMOTION` | `en` | 1.90s | 2026-11-26 | Fresh queen on the board! |
| `sick_ken/heavy_capture.wav` | `sick_ken` | `HEAVY_CAPTURE` | `en` | 1.21s | 2026-11-26 | Heavy piece off the board! |
| `focus_ken/heavy_capture.wav` | `focus_ken` | `HEAVY_CAPTURE` | `en` | 1.14s | 2026-11-26 | Heavy piece off the board! |
| `condition_ken/heavy_capture.wav` | `condition_ken` | `HEAVY_CAPTURE` | `en` | 1.16s | 2026-11-26 | Heavy piece off the board! |
| `sick_ken/castle.wav` | `sick_ken` | `CASTLE` | `en` | 1.69s | 2026-11-26 | King safely tucked away. |
| `focus_ken/castle.wav` | `focus_ken` | `CASTLE` | `en` | 1.58s | 2026-11-26 | King safely tucked away. |
| `condition_ken/castle.wav` | `condition_ken` | `CASTLE` | `en` | 1.62s | 2026-11-26 | King safely tucked away. |
| `sick_ken/draw.wav` | `sick_ken` | `DRAW` | `en` | 1.94s | 2026-11-26 | Good game! Solid draw. |
| `focus_ken/draw.wav` | `focus_ken` | `DRAW` | `en` | 1.81s | 2026-11-26 | Good game! Solid draw. |
| `condition_ken/draw.wav` | `condition_ken` | `DRAW` | `en` | 1.85s | 2026-11-26 | Good game! Solid draw. |
| `sick_ken/bot_loss.wav` | `sick_ken` | `LOSS` | `en` | 2.12s | 2026-11-26 | Nice game! You got me that time. |
| `focus_ken/bot_loss.wav` | `focus_ken` | `LOSS` | `en` | 1.98s | 2026-11-26 | Nice game! You got me that time. |
| `condition_ken/bot_loss.wav` | `condition_ken` | `LOSS` | `en` | 2.01s | 2026-11-26 | Nice game! You got me that time. |
| `sick_ken/bot_win.wav` | `sick_ken` | `WIN` | `en` | 1.81s | 2026-11-26 | GG! Want a rematch? |
| `focus_ken/bot_win.wav` | `focus_ken` | `WIN` | `en` | 1.69s | 2026-11-26 | GG! Want a rematch? |
| `condition_ken/bot_win.wav` | `condition_ken` | `WIN` | `en` | 1.73s | 2026-11-26 | GG! Want a rematch? |
