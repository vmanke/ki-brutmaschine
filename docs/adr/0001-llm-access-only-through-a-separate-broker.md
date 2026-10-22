# LLM access only through a separate Broker process

The core crate stays free of networking: it gains no HTTP dependency and never sees the API key. All calls to the language-model API go through a separate Broker (its own Cargo package), which alone reads the key, enforces the model allow-list and the spending caps, and records usage. We chose this over adding an HTTP client to the core because it keeps the core's dependency and attack surface unchanged, puts every cost and key control in one place, and makes it impossible for search or sandbox code to reach the network by accident.
