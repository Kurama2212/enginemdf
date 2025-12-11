# enginemdf

`enginemdf` is a lightweight, Python‑first interface designed to simplify the exploration and analysis of **AVL CONCERTO–extracted MDF files**.  
The goal is to provide a clean, intuitive API on top of `asammdf`, optimized for real automotive datasets containing long time‑series, many sensors, and multi‑segment acquisitions.

This project is currently in **early development** and serves both as an open‑source learning exercise and as a portfolio project.

---

## 🚧 Current Status (WIP)

- Basic project structure defined (`io`, `core`, `utils`, etc.)
- First implementation of **RawSegmentInfo**, a data structure describing MDF logical segments
- Early prototype of the I/O layer that:
  - Parses MDF file structure
  - Exposes segment metadata
  - Supports lazy loading (loader functions return only the requested segment)

Many components are still placeholders and the public API is not stable.

---

## 🎯 Objectives

- Provide a **simple, pythonic API** for working with MDF files
- Keep the library **fast and memory‑efficient**, suitable for very large measurements
- Improve usability beyond `asammdf` by:
  - Exposing both *physical channels* and *logical segments*
  - Allowing partial loading, chunking, lazy access
  - Offering dataframe‑friendly output
- Serve as a foundation for future analysis tools (e.g. signal processing, event handling)

---

## 📝 Roadmap / TODO

### I/O Layer
- [ ] Implement the MDF reader wrapper with:
  - [ ] Complete segment discovery
  - [ ] Proper mapping between MDF groups/channels and logical paths
  - [ ] Efficient lazy data extraction
- [ ] Add caching for frequently accessed channels

### Core API
- [ ] Define a clean `EngineMDF` class that:
  - [ ] Loads metadata once
  - [ ] Provides a stable API to access channels, segments, ranges
- [ ] Provide dataframe export utilities (Pandas + Arrow)

### Performance
- [ ] Benchmark loading strategies on real automotive datasets
- [ ] Introduce optional zero‑copy memory access where possible

### Documentation
- [ ] Add usage examples
- [ ] Add architecture overview
- [ ] Add contributor guidelines

---

## 🧪 Example (Preview)

```python
from enginemdf import EngineMDF

mdf = EngineMDF("path/to/file.mdf")

# List logical segments
for seg in mdf.segments:
    print(seg.measurement_name, seg.channel_name, seg.n_samples)

# Load data lazily
t, x = seg.loader()
```

> ⚠️ API not final — expect breaking changes.

---

## 🤝 Contributing

This is a learning-driven project, contributions and feedback are welcome.  
If you're using MDF files in the automotive field and have feature suggestions, feel free to open an issue.

---

## License

MIT License
