import {ChevronDown, Search, SlidersHorizontal, X} from "lucide-react";
import {useMemo, useState} from "react";

import type {ClassDefinition, ViewerDetection} from "../types";

type BroadClass = "all" | "ship" | "aircraft" | "vehicle";

interface FilterPanelProps {
  classes: ClassDefinition[];
  detections: ViewerDetection[];
  confidence: number;
  broadClass: BroadClass;
  selectedClasses: Set<number>;
  onConfidenceChange: (value: number) => void;
  onBroadClassChange: (value: BroadClass) => void;
  onToggleClass: (classId: number) => void;
  onClearClasses: () => void;
}

const broadOptions: Array<{id: BroadClass; label: string}> = [
  {id: "all", label: "全部"},
  {id: "ship", label: "舰船"},
  {id: "aircraft", label: "飞机"},
  {id: "vehicle", label: "车辆"},
];

export function FilterPanel({
  classes,
  detections,
  confidence,
  broadClass,
  selectedClasses,
  onConfidenceChange,
  onBroadClassChange,
  onToggleClass,
  onClearClasses,
}: FilterPanelProps) {
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState(() => !window.matchMedia("(max-width: 900px)").matches);
  const counts = useMemo(() => {
    const result = new Map<number, number>();
    detections.forEach((item) => result.set(item.classId, (result.get(item.classId) ?? 0) + 1));
    return result;
  }, [detections]);
  const broadCounts = useMemo(() => ({
    all: detections.length,
    ship: detections.filter((item) => item.broadClass === "ship").length,
    aircraft: detections.filter((item) => item.broadClass === "aircraft").length,
    vehicle: detections.filter((item) => item.broadClass === "vehicle").length,
  }), [detections]);
  const visibleClasses = classes.filter((item) => `${item.id} ${item.name}`.toLowerCase().includes(query.toLowerCase()));

  return (
    <aside className="filter-panel" aria-label="检测结果筛选">
      <div className="panel-title">
        <div>
          <span>结果筛选</span>
          <strong>{detections.length}</strong>
        </div>
        <SlidersHorizontal className="filter-symbol" size={17} />
        <button
          type="button"
          className="filter-toggle"
          aria-expanded={expanded}
          aria-label={expanded ? "收起筛选" : "展开筛选"}
          onClick={() => setExpanded((value) => !value)}
        >
          <ChevronDown className={expanded ? "expanded" : ""} size={17} />
        </button>
      </div>

      <div className={`filter-content ${expanded ? "expanded" : ""}`}>
      <section className="filter-section">
        <label htmlFor="confidence">最低置信度</label>
        <div className="confidence-readout">
          <input
            id="confidence"
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={confidence}
            onChange={(event) => onConfidenceChange(Number(event.target.value))}
          />
          <output htmlFor="confidence">{Math.round(confidence * 100)}%</output>
        </div>
      </section>

      <section className="filter-section">
        <span className="field-label">大类</span>
        <div className="broad-grid">
          {broadOptions.map((option) => (
            <button
              type="button"
              key={option.id}
              className={broadClass === option.id ? "active" : ""}
              onClick={() => onBroadClassChange(option.id)}
            >
              <span>{option.label}</span><b>{broadCounts[option.id]}</b>
            </button>
          ))}
        </div>
      </section>

      <section className="class-section">
        <div className="class-heading">
          <span>细粒度类别</span>
          {selectedClasses.size > 0 && (
            <button type="button" onClick={onClearClasses} title="清除类别筛选">
              <X size={14} /> 清除
            </button>
          )}
        </div>
        <label className="search-field">
          <Search size={15} />
          <span className="sr-only">搜索类别</span>
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索 ID 或名称" />
        </label>
        <div className="class-list" role="listbox" aria-label="细粒度类别" aria-multiselectable="true">
          {visibleClasses.map((item) => {
            const selected = selectedClasses.has(item.id);
            return (
              <button
                type="button"
                role="option"
                aria-selected={selected}
                className={selected ? "selected" : ""}
                key={item.id}
                onClick={() => onToggleClass(item.id)}
              >
                <span className="class-id">{String(item.id).padStart(2, "0")}</span>
                <span className="class-name">{item.name}</span>
                <span className="class-count">{counts.get(item.id) ?? 0}</span>
              </button>
            );
          })}
        </div>
      </section>
      </div>
    </aside>
  );
}
