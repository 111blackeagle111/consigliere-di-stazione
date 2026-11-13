# Changelog

## 1.1.1 — 2026-11-13

- Quando NOAA o POTA non rispondono, l'app conserva l'ultimo dato valido se disponibile. I messaggi di errore mostrati all'utente sono generici e i dettagli tecnici restano nei log del server.
- I consigli AI scartano le frequenze prive di riscontro nei dati correnti. Se RAG o Ollama restituiscono una risposta malformata, l'app può continuare con il fallback.
- L'esportazione ADI converte i commenti in ASCII e rappresenta FT4 e PSK31 come sottomodi. Le note originali si conservano nel registro e nel CSV.
- Aggiornate le dipendenze Python e la versione di `actions/checkout` usata per le build.

## 1.1.0 — 2026-08-19

- Analisi AI dell'intero storico tramite aggregazioni SQLite, insieme a NOAA e POTA su tutte le bande.
- Fallback espliciti: swlbot RAG, Qwen locale senza RAG, regole deterministiche.
- Barriera deterministica che scarta frequenze e deduzioni operative non supportate dai dati live, con avvisi distinti per il fallback Qwen e per le regole locali dopo verifica AI.
- QTH operatore con coordinate Maidenhead, distanza, azimut e direzione degli spot POTA.
- Storico completo filtrabile e paginato, con modifica ed eliminazione dei QSO manuali.
- Esportazione del registro in CSV e ADIF e backup SQLite scaricabile.
- Cache temporanea NOAA/POTA con orario del dato, età della cache e recupero dell'ultimo valore valido in caso di errore.
- Gestione coerente degli orari: visualizzazione locale, colonne locale/UTC nel CSV e data UTC nell'esportazione ADIF, senza riscrivere i vecchi timestamp.
- Validazione server-side, protezione delle scritture cross-site e ascolto solo su localhost per impostazione predefinita.
- Compatibilità conservata con i database delle versioni precedenti.
- README, pagina dei requisiti, guida Windows, manuale e schermate aggiornati; download Windows documentato tramite GitHub Releases con verifica SHA-256.
- Workflow di release predisposto per compilare l'eseguibile Windows e pubblicare il checksum soltanto da un tag coerente con `VERSION`.
