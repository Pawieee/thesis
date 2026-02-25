import { C, FONT, FONT_MONO } from '../../constants/theme';
import { Ic } from '../icons/Icons';
import { Card, Badge, StatBox, Divider } from '../ui/Primitives';
import VoteDots from './VoteDots';
import ProbDistChart from '../charts/ProbDistChart';
import PGenuineGauge from '../charts/PGenuineGauge';
import PerSupportChart from '../charts/PerSupportChart';

export default function ProposedPanel({ data }) {
  const isG = data.prediction === 'GENUINE';
  return (
    <Card borderColor={isG?C.greenBorder:C.redBorder}>
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
              PROPOSED MODEL · 3× K=1 Majority Vote
            </div>
            <div style={{ fontFamily:FONT, fontSize:26, fontWeight:800, color:isG?C.green:C.red, lineHeight:1, letterSpacing:-0.5 }}>
              {data.prediction}
            </div>
          </div>
        </div>
        <div style={{ textAlign:'right' }}>
          <div style={{ fontFamily:FONT, fontSize:11, color:C.textSub, marginBottom:2 }}>Vote Confidence</div>
          <div style={{ fontFamily:FONT_MONO, fontSize:24, fontWeight:700, color:C.navy }}>{data.vote_confidence}%</div>
        </div>
      </div>

      <div style={{ padding:'18px 20px' }}>
        <div style={{ fontFamily:FONT, fontSize:12, fontWeight:600, color:C.textSub, marginBottom:10 }}>Individual K=1 Comparisons</div>
        <VoteDots perSupport={data.per_support} />

        <Divider label="Probability Scores" />
        <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:9, marginBottom:4 }}>
          <StatBox label="Avg P(Genuine)"  value={(data.avg_p_genuine * 100).toFixed(2)} mono valueColor={C.green} />
          <StatBox label="Avg P(Forged)"   value={(data.avg_p_forged * 100).toFixed(2)}  mono valueColor={C.red}   />
          <StatBox label="Threshold (θ)"   value={(data.threshold * 100).toFixed(2)}     mono valueColor={C.blue}  />
          <StatBox label="Vote Tally"      value={`${data.per_support?.filter(s=>s.prediction==='GENUINE').length ?? '—'}/3`} mono />
        </div>

        <Divider label="Score Visualization" />
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:16, marginBottom:4 }}>
          <ProbDistChart pGenuine={data.avg_p_genuine} pForged={data.avg_p_forged} threshold={data.threshold} title="Probability Distribution" />
          <PGenuineGauge pGenuine={data.avg_p_genuine} threshold={data.threshold} title="P(Genuine) vs Threshold" />
        </div>

        <Divider label="Per-Support Comparison" />
        <div style={{ marginBottom:4 }}>
          <PerSupportChart perSupport={data.per_support} threshold={data.threshold} title="Per-Support Scores" />
        </div>

        <Divider label="Detailed Results Table" />
        <div style={{ borderRadius:8, overflow:'hidden', border:`1px solid ${C.border}` }}>
          <table style={{ width:'100%', borderCollapse:'collapse' }}>
            <thead>
              <tr style={{ background:C.surfaceAlt }}>
                {['Reference','P(Genuine)','P(Forged)','Vote'].map(h => (
                  <th key={h} style={{ padding:'8px 14px', fontFamily:FONT, fontSize:11, fontWeight:600, color:C.textSub, textAlign:'left', borderBottom:`1px solid ${C.border}` }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.per_support.map((s,i) => (
                <tr key={i} style={{ borderBottom:`1px solid ${C.borderLight}`, background:i%2?C.surfaceAlt:C.surface }}>
                  <td style={{ padding:'9px 14px', fontFamily:FONT, fontWeight:600, fontSize:12, color:C.navy }}>{s.support}</td>
                  <td style={{ padding:'9px 14px', fontFamily:FONT_MONO, fontSize:12, color:C.green }}>{(s.p_genuine * 100).toFixed(2)}</td>
                  <td style={{ padding:'9px 14px', fontFamily:FONT_MONO, fontSize:12, color:C.red   }}>{(s.p_forged * 100).toFixed(2)}</td>
                  <td style={{ padding:'9px 14px' }}><Badge color={s.prediction==='GENUINE'?'green':'red'}>{s.prediction}</Badge></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <Divider label="Technical Pipeline" />
        <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:10 }}>
          {[
            { title:'Feature Extraction', rows:[
              ['Support Features','1024-d × 3'],
              ['Query Features','1024-d × 1'],
              ['Combined Input','2048-d'],
              ['Preprocessing','Otsu → Invert → Resize → Normalize'],
            ]},
            { title:'Metric Computation', rows:[
              ['Operation','concat [support, query]'],
              ['MLP Output','Logit → Sigmoid'],
              ['Final Output','P(Genuine) ∈ [0, 1]'],
              ['Decision Rule',`P(G) ≥ ${(data.threshold * 100).toFixed(2)} → GENUINE`],
            ]},
          ].map(({ title, rows }) => (
            <div key={title} style={{ background:C.surfaceAlt, border:`1px solid ${C.border}`, borderRadius:8, padding:'12px 14px' }}>
              <div style={{ fontFamily:FONT, fontSize:11, fontWeight:700, color:C.blue, marginBottom:8 }}>{title}</div>
              {rows.map(([k,v]) => (
                <div key={k} style={{ display:'flex', justifyContent:'space-between', padding:'3px 0', borderBottom:`1px solid ${C.borderLight}`, gap:8 }}>
                  <span style={{ fontFamily:FONT, fontSize:11, color:C.textSub, flexShrink:0 }}>{k}</span>
                  <span style={{ fontFamily:FONT_MONO, fontSize:11, color:C.textMid, fontWeight:500, textAlign:'right' }}>{v}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}