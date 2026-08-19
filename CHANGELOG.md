# Changelog

## Unreleased — 1.1.0-dev

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

La versione definitiva e la data verranno assegnate soltanto dopo il collaudo finale.
