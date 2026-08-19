# Changelog

## Unreleased — 1.1.0-dev

- Analisi AI dell'intero storico tramite aggregazioni SQLite, insieme a NOAA e POTA su tutte le bande.
- Fallback espliciti: swlbot RAG, Qwen locale senza RAG, regole deterministiche.
- QTH operatore con coordinate Maidenhead, distanza, azimut e direzione degli spot POTA.
- Storico completo filtrabile e paginato, con modifica ed eliminazione dei QSO manuali.
- Esportazione del registro in CSV e ADIF e backup SQLite scaricabile.
- Cache temporanea NOAA/POTA con recupero dell'ultimo dato valido in caso di errore.
- Validazione server-side, protezione delle scritture cross-site e ascolto solo su localhost per impostazione predefinita.
- Compatibilità conservata con i database delle versioni precedenti.

La versione definitiva e la data verranno assegnate soltanto dopo il collaudo finale.
