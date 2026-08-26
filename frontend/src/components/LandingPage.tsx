import {
  ArrowRight,
  Boxes,
  CarFront,
  ChevronRight,
  CircleCheck,
  Crosshair,
  Database,
  Layers3,
  Plane,
  ScanLine,
  ShieldCheck,
  Ship,
  Sparkles,
  Target,
} from "lucide-react";
import {motion, useReducedMotion, useScroll, useTransform} from "motion/react";
import {useEffect, useRef} from "react";

interface LandingPageProps { onEnter: () => void; }

const stages = [
  {icon: ScanLine, label: "影像接入", detail: "读取万级像素完整图幅", value: "10K × 10K"},
  {icon: Layers3, label: "智能切片", detail: "规则网格覆盖边界区域", value: "169 TILES"},
  {icon: Target, label: "全图映射", detail: "将局部检测还原至全图", value: "GLOBAL XYXY"},
  {icon: Crosshair, label: "全图抑制", detail: "跨切片消解重复候选框", value: "NMS 0.70"},
  {icon: ShieldCheck, label: "逐类校准", detail: "依据25类独立阈值决策", value: "25 CLASSES"},
];

const groups = [
  {icon: Ship, label: "舰船", value: 60},
  {icon: Plane, label: "飞机", value: 60},
  {icon: CarFront, label: "车辆", value: 24},
];

type CapabilityVisualKind = "lineage" | "viewport" | "artifacts" | "regression";

const capabilities: Array<{icon: typeof Layers3; title: string; copy: string; meta: string; className: string; visual: CapabilityVisualKind}> = [
  {icon: Layers3, title: "完整阶段留痕", copy: "映射、抑制与校准结果独立保存，任一目标均可回看来源。", meta: "4个阶段", className: "capability-wide", visual: "lineage"},
  {icon: Boxes, title: "万级图幅判读", copy: "在完整影像坐标系中缩放、筛选与定位高密度目标。", meta: "1亿像素", className: "capability-image", visual: "viewport"},
  {icon: Database, title: "检测信息汇总", copy: "检测结果、性能指标、处理耗时与方法配置集中呈现。", meta: "全程可追溯", className: "capability-data", visual: "artifacts"},
  {icon: CircleCheck, title: "多场景适应", copy: "面向复杂、密集与稀疏场景，保持稳定清晰的检测表现。", meta: "多场景覆盖", className: "capability-proof", visual: "regression"},
];

function CapabilityVisual({kind}: {kind: CapabilityVisualKind}) {
  if (kind === "lineage") {
    return (
      <div className="capability-visual lineage-visual" aria-hidden="true">
        <div className="lineage-stage"><b>01</b><div><strong>切片推理</strong><small>局部检测结果</small></div></div>
        <div className="lineage-stage"><b>02</b><div><strong>全图映射</strong><small>全图坐标还原</small></div></div>
        <div className="lineage-stage"><b>03</b><div><strong>逐类校准</strong><small>最终检测结果</small></div></div>
      </div>
    );
  }

  if (kind === "viewport") {
    return (
      <div className="capability-visual viewport-visual" aria-hidden="true">
        <div className="viewport-orbit viewport-orbit-a" /><div className="viewport-orbit viewport-orbit-b" />
        <div className="viewport-window">
          <img src="./demo/demo_mosaic_preview.jpg" alt="" />
          <i className="viewport-target viewport-target-a" /><i className="viewport-target viewport-target-b" />
          <span className="viewport-crosshair" />
        </div>
        <div className="viewport-readout"><span>X 07342.8</span><span>Y 04186.3</span><b>100%</b></div>
      </div>
    );
  }

  if (kind === "artifacts") {
    return (
      <div className="capability-visual artifact-visual" aria-hidden="true">
        {[["01", "目标检测结果"], ["02", "综合性能指标"], ["03", "全流程耗时"]].map(([index, name]) => (
          <div className="artifact-row" key={name}><b>{index}</b><span>{name}</span><small>完整记录</small></div>
        ))}
      </div>
    );
  }

  return (
    <div className="capability-visual regression-visual" aria-hidden="true">
      {["复杂密集场景", "多类别目标场景", "稀疏目标场景"].map((label) => (
        <div className="regression-row" key={label}><CircleCheck size={17} /><span>{label}</span><small>稳定检测</small></div>
      ))}
    </div>
  );
}

export function LandingPage({onEnter}: LandingPageProps) {
  const pageRef = useRef<HTMLElement>(null);
  const reduceMotion = useReducedMotion();
  const {scrollYProgress} = useScroll({target: pageRef, offset: ["start start", "end end"]});
  const progressScale = useTransform(scrollYProgress, [0, 1], [0, 1]);
  const imageY = useTransform(scrollYProgress, [0, 0.45], [0, reduceMotion ? 0 : 42]);

  useEffect(() => {
    const sectionId = decodeURIComponent(window.location.hash.slice(1));
    if (!sectionId) return;
    window.requestAnimationFrame(() => document.getElementById(sectionId)?.scrollIntoView());
  }, []);

  return (
    <main ref={pageRef} className="landing-page promo-page">
      <motion.div className="promo-progress" style={{scaleX: progressScale}} />

      <nav className="landing-nav promo-nav" aria-label="平台导航">
        <button type="button" className="landing-brand" onClick={() => window.scrollTo({top: 0, behavior: reduceMotion ? "auto" : "smooth"})}>
          <span><Crosshair size={19} /></span>
          <div><strong>RS-CalVision</strong><small>遥感目标检测与追溯平台</small></div>
        </button>
        <div className="landing-nav-links">
          <a href="#技术链路">技术链路</a>
          <a href="#平台能力">平台能力</a>
          <button type="button" onClick={onEnter}><span>进入工作台</span><ArrowRight size={16} /></button>
        </div>
      </nav>

      <section className="promo-hero">
        <div className="promo-hero-copy">
          <motion.h1
            initial={reduceMotion ? false : {opacity: 0, y: 28, filter: "blur(8px)"}}
            animate={{opacity: 1, y: 0, filter: "blur(0px)"}}
            transition={{duration: 0.72, ease: [0.16, 1, 0.3, 1]}}
          >
            让万级影像，<br /><em>逐层说清结果</em>
          </motion.h1>
          <motion.p
            initial={reduceMotion ? false : {opacity: 0, y: 16}}
            animate={{opacity: 1, y: 0}}
            transition={{duration: 0.55, delay: 0.16}}
          >
            面向超大幅面光学遥感影像，把切片推理、全图映射、重复抑制与逐类校准组织为一条可核验链路。
          </motion.p>
          <motion.div
            className="promo-actions"
            initial={reduceMotion ? false : {opacity: 0, y: 14}}
            animate={{opacity: 1, y: 0}}
            transition={{duration: 0.5, delay: 0.26}}
          >
            <button type="button" className="promo-primary" onClick={onEnter}>进入检测工作台 <ArrowRight size={18} /></button>
            <a className="promo-secondary" href="#技术链路">查看技术链路 <ChevronRight size={17} /></a>
          </motion.div>
        </div>

        <motion.div
          className="survey-visual"
          style={{y: imageY}}
          initial={reduceMotion ? false : {opacity: 0, clipPath: "inset(8% 0 8% 100%)"}}
          animate={{opacity: 1, clipPath: "inset(0% 0 0% 0%)"}}
          transition={{duration: 0.92, delay: 0.08, ease: [0.16, 1, 0.3, 1]}}
        >
          <div className="survey-visual-head">
            <div><span className="visual-status" /><strong>海陆空多目标演示图</strong></div>
            <span>10000 × 10000 PX</span>
          </div>
          <div className="survey-image-frame">
            <img src="./demo/demo_mosaic_preview.jpg" alt="由舰船、飞机和车辆裁剪图构成的多类别遥感演示大图" />
            <div className="survey-grid" aria-hidden="true" />
            <div className="survey-scan" aria-hidden="true" />
            <i className="demo-box box-a" /><i className="demo-box box-b" /><i className="demo-box box-c" />
            <div className="coordinate-readout"><span>X 07342.8</span><span>Y 04186.3</span></div>
          </div>
          <div className="survey-visual-foot">
            <span>144 张源影像</span><span>600 个标注目标</span><span>800 px 对齐网格</span>
          </div>
        </motion.div>

        <motion.div
          className="evidence-rail"
          initial={reduceMotion ? false : {opacity: 0, y: 18}}
          animate={{opacity: 1, y: 0}}
          transition={{duration: 0.55, delay: 0.4}}
        >
          <div className="evidence-context"><Sparkles size={18} /><div><strong>高精度目标检测表现</strong><span>RS-TailCalDet · 25类目标识别</span></div></div>
          <dl>
            <div><dt>召回率</dt><dd>98.21<small>%</small></dd></div>
            <div><dt>虚警率</dt><dd>3.51<small>%</small></dd></div>
          </dl>
          <p>支持舰船、飞机与车辆等多类别目标的统一检测、定位与结果判读。</p>
        </motion.div>
      </section>

      <section className="source-ribbon" aria-label="演示影像类别构成">
        <div className="source-ribbon-title"><strong>海陆空目标覆盖</strong><span>多源遥感样例组合展示</span></div>
        {groups.map(({icon: Icon, label, value}) => (
          <div className="source-group" key={label}><Icon size={20} /><span>{label}</span><b>{value}</b><small>张源影像</small></div>
        ))}
      </section>

      <section className="route-section" id="技术链路">
        <motion.div className="section-intro" initial={reduceMotion ? false : {opacity: 0, y: 26}} whileInView={{opacity: 1, y: 0}} viewport={{once: true, amount: 0.45}} transition={{duration: 0.55}}>
          <h2>一条链路，贯穿整幅影像</h2>
          <p>所有阶段共享同一运行身份和全图坐标系，切换结果时无需重新推理。</p>
        </motion.div>
        <div className="route-track">
          <motion.div className="route-beam" initial={{scaleX: 0}} whileInView={{scaleX: 1}} viewport={{once: true}} transition={{duration: 1.1, ease: [0.16, 1, 0.3, 1]}} />
          {stages.map(({icon: Icon, label, detail, value}, index) => (
            <motion.article className="route-stage" key={label} initial={reduceMotion ? false : {opacity: 0, y: 24}} whileInView={{opacity: 1, y: 0}} viewport={{once: true, amount: 0.5}} transition={{duration: 0.45, delay: index * 0.08}}>
              <div className="route-icon"><Icon size={21} /></div>
              <strong>{label}</strong><p>{detail}</p><small>{value}</small>
            </motion.article>
          ))}
        </div>
      </section>

      <section className="capability-section" id="平台能力">
        <div className="section-intro"><h2>为科研复现而生，不止于可视化</h2><p>界面服务于方法身份、阶段产物和评价边界的统一表达。</p></div>
        <div className="capability-grid">
          {capabilities.map(({icon: Icon, title, copy, meta, className, visual}, index) => (
            <motion.article className={`capability-cell ${className}`} key={title} initial={reduceMotion ? false : {opacity: 0, y: 22}} whileInView={{opacity: 1, y: 0}} viewport={{once: true, amount: 0.35}} transition={{duration: 0.48, delay: index * 0.07}}>
              <Icon size={27} /><span>{meta}</span><CapabilityVisual kind={visual} /><div className="capability-copy"><h3>{title}</h3><p>{copy}</p></div>
            </motion.article>
          ))}
        </div>
      </section>

      <section className="promo-closing" id="数据说明">
        <div><h2>从影像到结论，每一步都有依据</h2><p>进入工作台查看目标坐标、来源切片、阶段结果、评价指标与运行耗时。</p></div>
        <button type="button" className="promo-primary" onClick={onEnter}>打开 RS-CalVision <ArrowRight size={18} /></button>
      </section>

      <footer className="promo-footer"><strong>RS-CalVision</strong><span>超大幅面光学遥感目标检测与追溯平台</span><small>舰船 · 飞机 · 车辆多类别覆盖</small></footer>
    </main>
  );
}
