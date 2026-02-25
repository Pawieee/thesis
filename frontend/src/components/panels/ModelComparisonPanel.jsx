import { C, FONT, FONT_MONO } from '../../constants/theme';
import { Ic } from '../icons/Icons';
import { Card, SectionHeader, Divider } from '../ui/Primitives';

export default function ModelComparisonPanel({ proposed, baseline }) {
  const speedup = (proposed.processing_time / baseline.processing_time).toFixed(1);
  const proposedIsG = proposed.prediction === 'GENUINE';
  const baselineIsG = baseline.prediction === 'GENUINE';
  const agree = proposed.prediction === baseline.prediction;

  return (
    <Card>
      <SectionHeader icon={Ic.chart} title="Model Comparison" subtitle="Proposed (tDCBAM) vs Baseline (DenseNet Classifier)" />
      <div style={{ padding:'18px 20px' }}>
        <div style={{
          background: agree ? C.greenLight : C.amberLight,
          border: `1px solid ${agree ? C.greenBorder : C.amberBorder}`,
          borderRadius: 9, padding:'10px 16px',
          display:'flex', alignItems:'center', gap:10, marginBottom:18,
        }}>
          <span style={{ width:16, height:16, color: agree ? C.green : C.amber, flexShrink:0 }}>
            {agree ? Ic.check : Ic.alert}
          </span>
          <span style={{ fontFamily:FONT, fontSize:13, fontWeight:600, color: agree ? C.green : '#78350F' }}>
            {agree
              ? `Both models agree — verdict is ${proposed.prediction}`
              : `Models disagree — Proposed: ${proposed.prediction} · Baseline: ${baseline.prediction}`}
          </span>
        </div>

        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:12, marginBottom:18 }}>
          {[
            { label:'Proposed · tDCBAM',    prediction: proposed.prediction, time: proposed.processing_time, color: proposedIsG ? C.green : C.red, detail:'3× K=1 Majority Vote · 4 images' },
            { label:'Baseline · DenseNet',  prediction: baseline.prediction, time: baseline.processing_time, color: baselineIsG ? C.green : C.red, detail:'Single query · no support set' },
          ].map(({ label, prediction, time, color, detail }) => (
            <div key={label} style={{ background:C.surfaceAlt, border:`1px solid ${C.border}`, borderRadius:10, padding:'14px 16px' }}>
              <div style={{ fontFamily:FONT, fontSize:11, fontWeight:600, color:C.textSub, marginBottom:6 }}>{label}</div>
              <div style={{ fontFamily:FONT, fontSize:20, fontWeight:800, color, marginBottom:6 }}>{prediction}</div>
              <div style={{ display:'flex', alignItems:'center', gap:6 }}>
                <span style={{ width:12, height:12, color:C.textFaint, display:'block' }}>{Ic.clock}</span>
                <span style={{ fontFamily:FONT_MONO, fontSize:12, color:C.textMid }}>{time}s</span>
              </div>
              <div style={{ fontFamily:FONT, fontSize:11, color:C.textFaint, marginTop:4 }}>{detail}</div>
            </div>
          ))}
        </div>

        <Divider label="Score Comparison" />
        <div style={{ borderRadius:8, overflow:'hidden', border:`1px solid ${C.border}`, marginBottom:18 }}>
          <table style={{ width:'100%', borderCollapse:'collapse' }}>
            <thead>
              <tr style={{ background:C.surfaceAlt }}>
                {['Metric','Proposed','Baseline'].map(h => (
                  <th key={h} style={{ padding:'8px 14px', fontFamily:FONT, fontSize:11, fontWeight:600, color:C.textSub, textAlign:'left', borderBottom:`1px solid ${C.border}` }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                ['Verdict',       proposed.prediction,                              baseline.prediction],
                ['P(Genuine)',    (proposed.avg_p_genuine * 100).toFixed(2),        (baseline.p_genuine * 100).toFixed(2)],
                ['P(Forged)',     (proposed.avg_p_forged * 100).toFixed(2),         (baseline.p_forged * 100).toFixed(2)],
                ['Confidence',   `${proposed.vote_confidence}%`,                   `${baseline.confidence.toFixed(2)}%`],
                ['Threshold θ',  (proposed.threshold * 100).toFixed(2),            (baseline.threshold * 100).toFixed(2)],
                ['Process Time', `${proposed.processing_time}s`,                   `${baseline.processing_time}s`],
              ].map(([label, pVal, bVal], i) => (
                <tr key={label} style={{ borderBottom:`1px solid ${C.borderLight}`, background:i%2?C.surfaceAlt:C.surface }}>
                  <td style={{ padding:'9px 14px', fontFamily:FONT, fontWeight:500, fontSize:12, color:C.navy }}>{label}</td>
                  <td style={{ padding:'9px 14px', fontFamily:FONT_MONO, fontSize:12, color:C.textMid }}>{pVal}</td>
                  <td style={{ padding:'9px 14px', fontFamily:FONT_MONO, fontSize:12, color:C.textMid }}>{bVal}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <Divider label="Speed vs Accuracy Trade-off" />
        <div style={{ background:C.amberLight, border:`1px solid ${C.amberBorder}`, borderRadius:8, padding:'12px 16px', fontFamily:FONT, fontSize:12, color:'#78350F', lineHeight:1.65 }}>
          ⚡ The Baseline is <strong>{speedup}× faster</strong> ({baseline.processing_time}s vs {proposed.processing_time}s).
          The Proposed model processes <strong>4 images</strong> (3 supports + 1 query) using metric learning for more robust verification,
          at the cost of higher inference time. The Baseline classifies the query image alone with no reference set.
        </div>
      </div>
    </Card>
  );
}