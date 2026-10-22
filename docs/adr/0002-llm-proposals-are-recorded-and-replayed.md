# LLM proposals are recorded and replayed

Language-model answers are not reproducible, but every Run must be ("same configuration, same archive"). So every request to the LLM proposer and its full answer are stored in the Run's records, and a repeated Run replays the stored answers instead of calling the API again. The alternative, accepting non-reproducible LLM Runs, was rejected: it would break the determinism checks the whole project relies on and would charge the API again for every re-run.
