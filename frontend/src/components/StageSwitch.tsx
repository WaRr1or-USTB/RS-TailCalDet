import type {ViewStage} from "../types";

const stages: Array<{id: ViewStage; label: string}> = [
  {id: "final", label: "最终结果"},
  {id: "after_global_nms", label: "全图抑制后"},
  {id: "mapped", label: "全图映射"},
  {id: "ground_truth", label: "标注真值"},
];

export function StageSwitch({value, onChange, gtAvailable}: {value: ViewStage; onChange: (stage: ViewStage) => void; gtAvailable: boolean}) {
  return (
    <div className="stage-switch" role="tablist" aria-label="检测阶段">
      {stages.map((stage) => (
        <button
          type="button"
          role="tab"
          aria-selected={value === stage.id}
          disabled={stage.id === "ground_truth" && !gtAvailable}
          className={value === stage.id ? "active" : ""}
          key={stage.id}
          onClick={() => onChange(stage.id)}
        >
          {stage.label}
        </button>
      ))}
    </div>
  );
}
