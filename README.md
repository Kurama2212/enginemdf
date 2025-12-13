# enginemdf

`enginemdf` is a lightweight, Python-first interface designed to simplify the exploration and analysis of **AVL CONCERTO–extracted MDF files**.  
The goal is to provide a clean, intuitive API on top of `asammdf`, optimized for real automotive datasets containing long time-series, many sensors, and multi-segment acquisitions.

This project is currently in **early development** and serves both as an open-source learning exercise and as a portfolio project.

---

## 🚧 Current Status (WIP)

The project has moved beyond a pure prototype and now includes a first stable **core layer**.

### Implemented
- Clear project structure (`io`, `core`, `utils`, `test`)
- **Core abstractions**:
  - `TimeSeries`: immutable time base with slicing, masking and consistency checks
  - `Channel`: logical signal abstraction combining data, time axis and metadata
- Strong validation and explicit error handling
- First unit test suite with `pytest`
- Early MDF parsing utilities and segment metadata discovery

### In Progress
- Integration between MDF segments and `Channel` / `TimeSeries`
- Stabilization of the public API
- Performance tuning on real-world automotive datasets

The API is still evolving and **breaking changes are expected**.

---

## 🧠 Design Principles

- **Explicit over implicit**: no hidden assumptions on time bases or alignment
- **Metadata-first**: structure and meaning before raw arrays
- **Lazy but deterministic**: data is loaded on demand, behavior is predictable
- **Automotive-oriented**: designed around real measurement workflows, not generic time-series

---

## 🎯 Objectives

- Provide a **simple, pythonic API** for working with MDF files
- Keep the library **fast and memory-efficient**, suitable for very large measurements
- Improve usability beyond `asammdf` by:
  - Exposing both *physical channels* and *logical segments*
  - Allowing partial loading, chunking and lazy access
  - Offering dataframe-friendly output
- Act as a foundation for higher-level analysis tools
  (signal processing, data quality checks, ML-based diagnostics)

---

## 📝 Roadmap / TODO

### I/O Layer
- [x] Complete MDF reader wrapper
- [x] Robust segment discovery and classification
- [x] Mapping between MDF groups/channels and logical paths
- [ ] Efficient lazy data extraction
- [ ] Optional caching for frequently accessed channels

### Core API
- [ ] Finalize `EngineMDF` high-level interface
- [ ] Tight integration between MDF data and `Channel`
- [ ] Range-based and event-based slicing utilities
- [ ] DataFrame / Arrow export helpers

### Data Quality & Analysis (Future)
- [ ] Metadata channel detection and separation
- [ ] Rule-based data quality checks
- [ ] Hooks for anomaly detection and ML-based validation

### Performance
- [ ] Benchmark on long acquisitions (hours of data)
- [ ] Evaluate zero-copy / memory-mapped strategies

### Documentation
- [ ] Usage examples
- [ ] Core concepts documentation
- [ ] Architecture overview
- [ ] Contributor guidelines

---

## 🧪 Example (Preview)

```python
from enginemdf.core import Channel

rpm = Channel(
    name="engine_speed",
    time=t,
    data=x,
    unit="rpm"
)

# Slice in time
rpm_high = rpm.time_slice(t_min=10.0, t_max=20.0)

# Access raw arrays
t, x = rpm_high.time, rpm_high.data
```

> ⚠️ API not final — expect breaking changes.

---

## 🤝 Contributing

This is a learning-driven project, but feedback and contributions are welcome.  
If you work with MDF files in the automotive field and have suggestions, feel free to open an issue.

---

## License

MIT License
