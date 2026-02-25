import { C, FONT, FONT_MONO } from '../../constants/theme';
import { Ic } from '../icons/Icons';
import { Card, StatBox, Divider } from '../ui/Primitives';
import ProbDistChart from '../charts/ProbDistChart';
import PGenuineGauge from '../charts/PGenuineGauge';

export default function BaselinePanel({ data, proposedData }) {
  const isG = data.prediction === 'GENUINE';
  return (
    <Card>
      <div style={{
        padding:'16px 20px',
        background:isG?C.greenLight:C.redLight,
        borderBottom:`1px solid ${isG?C.greenBorder:C.redBorder}`,
        display:'flex', alignItems:'center', justifyContent:'space-between',
      }}>
        <div style={{ display:'flex', alignItems:'center', gap:12 }}>
          <span style={{ width:24, height:24, color:isG?C.green:C.red }}>{isG?Ic.check:Ic.alert}</span>
          <div>
            <div style={{ fontFamily:FONT, fontSize:11, fontWeight:600, color:isG?C.green:C.red, letterSpacing:0.3, marginBottom:2 }}>
              BASELINE MODEL · DenseNet Classifier
            </div>
            <div style={{ fontFamily:FONT, fontSize:26, fontWeight:800, color:isG?C.green:C.red, lineHeight:1, letterSpacing:-0.5 }}>
              {data.prediction}
            </div>
          </div>
        </div>
        <div style={{ textAlign:'right' }}>
          <div style={{ fontFamily:FONT, fontSize:11, color:isG?C.green:C.red, marginBottom:2 }}>Confidence</div>
          <div style={{ fontFamily:FONT_MONO, fontSize:24, fontWeight:700, color:isG?C.green:C.red }}>{data.confidence.toFixed(2)}%</div>
        </div>
      </div>

      <div style={{ padding:'18px 20px' }}>
        <div style={{ background:C.blueLight, border:`1px solid ${C.border}`, borderRadius:8, padding:'10px 14px', fontFamily:FONT, fontSize:12, color:C.textMid, marginBottom:16, lineHeight:1.55 }}>
          The baseline DenseNet classifier is evaluated on the <strong>query image only</strong> — it does not use the 3 support references.
        </div>

        <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:9, marginBottom:4 }}>
          <StatBox label="Verdict"    value={data.prediction}                        mono={false} valueColor={isG?C.green:C.red} />
          <StatBox label="P(Genuine)" value={(data.p_genuine * 100).toFixed(2)}       mono valueColor={C.green} />
          <StatBox label="P(Forged)"  value={(data.p_forged * 100).toFixed(2)}        mono valueColor={C.red}   />
          <StatBox label="Confidence" value={(data.confidence).toFixed(2) + '%'}      mono />
        </div>

        <Divider label="Score Visualization" />
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:16, marginBottom:4 }}>
          <ProbDistChart pGenuine={data.p_genuine} pForged={data.p_forged} threshold={data.threshold} title="Baseline Probability Distribution" />
          <PGenuineGauge pGenuine={data.p_genuine} threshold={data.threshold} title="Baseline P(Genuine) vs Threshold" />
        </div>

        <Divider label="Probability Breakdown" />
        <div style={{ borderRadius:8, overflow:'hidden', border:`1px solid ${C.border}` }}>
          <table style={{ width:'100%', borderCollapse:'collapse' }}>
            <thead>
              <tr style={{ background:C.surfaceAlt }}>
                {['Metric','Value','vs Threshold'].map(h => (
                  <th key={h} style={{ padding:'8px 14px', fontFamily:FONT, fontSize:11, fontWeight:600, color:C.textSub, textAlign:'left', borderBottom:`1px solid ${C.border}` }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                ['P(Genuine)',   (data.p_genuine * 100).toFixed(2),   `${((data.p_genuine - data.threshold) * 100).toFixed(2)}`],
                ['P(Forged)',    (data.p_forged * 100).toFixed(2),    `${((data.p_forged - data.threshold) * 100).toFixed(2)}`],
                ['Threshold θ', (data.threshold * 100).toFixed(2),   '—'],
              ].map(([label,val,diff],i) => (
                <tr key={label} style={{ borderBottom:`1px solid ${C.borderLight}`, background:i%2?C.surfaceAlt:C.surface }}>
                  <td style={{ padding:'9px 14px', fontFamily:FONT, fontWeight:500, fontSize:12, color:C.navy }}>{label}</td>
                  <td style={{ padding:'9px 14px', fontFamily:FONT_MONO, fontSize:12, color:C.textMid }}>{val}</td>
                  <td style={{ padding:'9px 14px', fontFamily:FONT_MONO, fontSize:12, color:diff.startsWith('-')?C.red:(diff==='—'?C.textFaint:C.green) }}>{diff}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Card>
  );
}