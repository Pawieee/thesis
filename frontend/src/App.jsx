import React from 'react';
import './index.css'; // Your global styles

import { useVerification } from './hooks/useVerification';
import { C, FONT, FONT_MONO } from './constants/theme';
import { DATASETS, SPLITS } from './constants/config';
import { Ic } from './components/icons/Icons';

import { Card, SectionHeader, Badge, Divider, UploadCard } from './components/ui/Primitives';
import ProposedPanel from './components/panels/ProposedPanel';
import BaselinePanel from './components/panels/BaselinePanel';
import ModelComparisonPanel from './components/panels/ModelComparisonPanel';

export default function App() {
  const {
    dataset, setDataset, split, setSplit,
    supports, previews, handleSupport, clearSupport,
    queryFile, queryPreview, handleQuery, clearQuery,
    loading, result, error, resultsRef, allReady, verify
  } = useVerification();

  const proposed = result?.proposed;
  const baseline = result?.baseline;

  return (
    <div style={{ width: '100%', minHeight: '100vh', background: C.bg }}>

      {/* ── Sticky Nav ──────────────────────────────────────────────── */}
      <header style={{
        width: '100%', background: C.navy, borderBottom: `1px solid ${C.navyMid}`,
        padding: '0 32px', display: 'flex', alignItems: 'center', height: 58, gap: 14,
        position: 'sticky', top: 0, zIndex: 100, boxShadow: '0 2px 12px rgba(0,0,0,0.20)',
      }}>
        <span style={{ width: 22, height: 22, color: '#60A5FA', flexShrink: 0 }}>{Ic.shield}</span>
        <span style={{ fontFamily: FONT, fontWeight: 700, fontSize: 17, color: '#F1F5F9', letterSpacing: -0.3 }}>
          Signature Verification
        </span>
        <span style={{ fontFamily: FONT_MONO, fontSize: 11, color: '#60A5FA' }}>tDCBAM · 3×K=1</span>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 7 }}>
          <div style={{ width: 7, height: 7, borderRadius: '50%', background: '#4ADE80', boxShadow: '0 0 6px #4ADE80' }} />
          <span style={{ fontFamily: FONT, fontSize: 12, color: '#94A3B8' }}>API Online</span>
        </div>
      </header>

      {/* ── Main Content — full width ─────────────────────────────── */}
      <div style={{ width: '100%', padding: '28px 32px 56px' }}>

        {/* Page heading */}
        <div style={{ marginBottom: 22 }}>
          <h1 style={{ fontFamily: FONT, fontWeight: 700, fontSize: 22, color: C.navy, letterSpacing: -0.4, marginBottom: 6 }}>
            Test Set Protocol — K=1 Episodic Evaluation
          </h1>
          <p style={{ fontFamily: FONT, fontSize: 13, color: C.textSub, lineHeight: 1.65, maxWidth: 820 }}>
            Upload <strong>3 genuine support signatures</strong> and <strong>1 query signature</strong>.
            The model performs 3 independent K=1 comparisons using a learned metric and returns a <strong>majority vote</strong> as the final verdict.
            The learned decision threshold minimises the Equal Error Rate on the validation set.
          </p>
        </div>

        {/* ── Config + Upload row ─────────────────────────────────── */}
        <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: 18, marginBottom: 20, alignItems: 'start' }}>

          {/* Left: Config sidebar */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <Card>
              <SectionHeader icon={Ic.settings} title="Configuration" />
              <div style={{ padding: '16px' }}>
                <div style={{ marginBottom: 14 }}>
                  <label style={{ display: 'block', fontFamily: FONT, fontWeight: 600, fontSize: 12, color: C.textMid, marginBottom: 6 }}>Dataset</label>
                  <div style={{ position: 'relative' }}>
                    <select value={dataset} onChange={e => setDataset(e.target.value)} style={{ width: '100%', padding: '9px 34px 9px 11px', background: C.surfaceAlt, border: `1px solid ${C.border}`, borderRadius: 8, fontFamily: FONT, fontSize: 13, color: C.text, cursor: 'pointer', outline: 'none' }}>
                      {DATASETS.map(d => <option key={d.value} value={d.value}>{d.label}</option>)}
                    </select>
                    <span style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', color: C.textFaint, pointerEvents: 'none' }}>▾</span>
                  </div>
                </div>
                <div>
                  <label style={{ display: 'block', fontFamily: FONT, fontWeight: 600, fontSize: 12, color: C.textMid, marginBottom: 7 }}>Train / Val / Test Split</label>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {SPLITS.map(s => (
                      <div key={s.value} onClick={() => setSplit(s.value)} style={{
                        padding: '8px 12px', borderRadius: 8,
                        border: `1.5px solid ${split === s.value ? C.blue : C.border}`,
                        background: split === s.value ? C.blueLight : C.surfaceAlt,
                        cursor: 'pointer', transition: 'all 0.15s',
                        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                      }}>
                        <span style={{ fontFamily: FONT_MONO, fontSize: 13, fontWeight: 600, color: split === s.value ? C.blue : C.textMid }}>{s.label}</span>
                        {split === s.value && <Badge color="blue">Active</Badge>}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </Card>

            <Card>
              <SectionHeader icon={Ic.layers} title="Architecture" />
              <div style={{ padding: '12px 16px' }}>
                {[
                  ['Backbone', 'DenseNet-121 + CBAM'],
                  ['Feat. Dim', '1024-d embeddings'],
                  ['Metric MLP', '2048 → 512 → 1'],
                  ['Activation', 'Sigmoid'],
                  ['Loss Fn', 'BCEWithLogitsLoss'],
                  ['Threshold', 'Learned (min. EER)'],
                  ['Final Vote', '≥ 2/3 → GENUINE'],
                ].map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', padding: '5px 0', borderBottom: `1px solid ${C.borderLight}` }}>
                    <span style={{ fontFamily: FONT, fontSize: 12, color: C.textSub }}>{k}</span>
                    <span style={{ fontFamily: FONT_MONO, fontSize: 11, color: C.textMid, fontWeight: 500, textAlign: 'right', maxWidth: 130 }}>{v}</span>
                  </div>
                ))}
              </div>
            </Card>
          </div>

          {/* Right: Upload panel */}
          <Card>
            <SectionHeader icon={Ic.upload} title="Upload Signatures" subtitle="3 genuine supports + 1 query to verify" />
            <div style={{ padding: '20px' }}>

              {/* Supports */}
              <div style={{ marginBottom: 18 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                  <Badge color="amber">Support Set · 3× K=1</Badge>
                  <span style={{ fontFamily: FONT, fontSize: 12, color: C.textSub }}>Genuine reference signatures from the test user</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 14 }}>
                  {[0, 1, 2].map(i => (
                    <UploadCard key={i} label={`Support ${i + 1}`} sublabel="Genuine Ref" file={supports[i]} preview={previews[i]} onChange={handleSupport(i)} onClear={clearSupport(i)} accent="amber" />
                  ))}
                </div>
              </div>

              <Divider label="Query Signature" />

              {/* Query Signature + Status + CTA stacked */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

                {/* Query upload — full width */}
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                    <Badge color="red">Query</Badge>
                    <span style={{ fontFamily: FONT, fontSize: 12, color: C.textSub }}>Genuine or forged</span>
                  </div>
                  <UploadCard label="Query Signature" sublabel="Test Image" file={queryFile} preview={queryPreview} onChange={handleQuery} onClear={clearQuery} accent="rose" />
                </div>

                {/* Upload status grid */}
                <div style={{ background: C.surfaceAlt, border: `1px solid ${C.border}`, borderRadius: 10, padding: '12px 16px' }}>
                  <div style={{ fontFamily: FONT, fontSize: 12, fontWeight: 600, color: C.textMid, marginBottom: 10 }}>Upload Status</div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 8 }}>
                    {[
                      { label: 'Support 1', ready: !!supports[0] },
                      { label: 'Support 2', ready: !!supports[1] },
                      { label: 'Support 3', ready: !!supports[2] },
                      { label: 'Query', ready: !!queryFile },
                    ].map(({ label, ready }) => (
                      <div key={label} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 5 }}>
                        <div style={{
                          width: 34, height: 34, borderRadius: '50%',
                          background: ready ? C.greenLight : '#F3F4F6',
                          border: `2px solid ${ready ? C.greenBorder : C.border}`,
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                          color: ready ? C.green : C.textFaint, transition: 'all 0.2s',
                        }}>
                          {ready
                            ? <span style={{ width: 14, height: 14, display: 'block' }}>{Ic.check}</span>
                            : <span style={{ fontFamily: FONT_MONO, fontSize: 13, fontWeight: 700 }}>?</span>
                          }
                        </div>
                        <span style={{ fontFamily: FONT, fontSize: 11, color: ready ? C.green : C.textFaint, fontWeight: ready ? 600 : 400 }}>{label}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* CTA */}
                <button
                  onClick={verify}
                  disabled={loading || !allReady}
                  onMouseEnter={e => { if (!loading && allReady) e.currentTarget.style.background = C.blue; }}
                  onMouseLeave={e => { if (!loading && allReady) e.currentTarget.style.background = C.navy; }}
                  style={{
                    width: '100%', padding: '13px 20px',
                    background: loading || !allReady ? '#E5E7EB' : C.navy,
                    color: loading || !allReady ? C.textFaint : '#FFF',
                    border: 'none', borderRadius: 10,
                    fontFamily: FONT, fontWeight: 700, fontSize: 14,
                    cursor: loading || !allReady ? 'not-allowed' : 'pointer',
                    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
                    transition: 'background 0.18s ease',
                    boxShadow: loading || !allReady ? 'none' : '0 4px 14px rgba(15,30,54,0.24)',
                  }}
                >
                  {loading
                    ? <><span style={{ width: 17, height: 17, display: 'inline-block' }}>{Ic.loader}</span> Analyzing…</>
                    : <><span style={{ width: 17, height: 17, display: 'inline-block' }}>{Ic.shield}</span> Run Verification</>
                  }
                </button>

                {/* Error */}
                {error && (
                  <div style={{ background: C.redLight, border: `1px solid ${C.redBorder}`, borderRadius: 8, padding: '10px 14px', display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                    <span style={{ width: 15, height: 15, color: C.red, flexShrink: 0, marginTop: 1 }}>{Ic.alert}</span>
                    <div>
                      <div style={{ fontFamily: FONT, fontWeight: 600, fontSize: 12, color: C.red, marginBottom: 2 }}>Verification Failed</div>
                      <div style={{ fontFamily: FONT, fontSize: 12, color: '#7F1D1D' }}>{error}</div>
                    </div>
                  </div>
                )}

              </div>
            </div>
          </Card>
        </div>

        {/* ── Results ─────────────────────────────────────────────── */}
        {result && (
          <div ref={resultsRef} className="fade-up">
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
              <span style={{ width: 20, height: 20, color: C.blue }}>{Ic.chart}</span>
              <h2 style={{ fontFamily: FONT, fontWeight: 700, fontSize: 19, color: C.navy, letterSpacing: -0.3 }}>Verification Results</h2>
              <div style={{ flex: 1, height: 1, background: C.border }} />
              <Badge color={proposed.prediction === 'GENUINE' ? 'green' : 'red'}>Final: {proposed.prediction}</Badge>
            </div>

            {/* Side-by-side panels */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: baseline?.available ? '1fr 1fr' : '1fr',
              gap: 18,
              alignItems: 'start',
            }}>
              <ProposedPanel data={proposed} />
              {baseline?.available && <BaselinePanel data={baseline} proposedData={proposed} />}
            </div>

            {/* Model comparison container */}
            {baseline?.available && (
              <div style={{ marginTop: 18 }}>
                <ModelComparisonPanel proposed={proposed} baseline={baseline} />
              </div>
            )}

            {result.baseline && !baseline?.available && (
              <div style={{ marginTop: 12, background: C.surface, border: `1px solid ${C.border}`, borderRadius: 10, padding: '12px 16px', display: 'flex', gap: 8, alignItems: 'center' }}>
                <span style={{ width: 15, height: 15, color: C.amber }}>{Ic.info}</span>
                <span style={{ fontFamily: FONT, fontSize: 13, color: C.textSub }}>Baseline model is unavailable for the <strong>Combined</strong> dataset.</span>
              </div>
            )}
          </div>
        )}

        {/* Footer */}
        <div style={{ marginTop: 40, borderTop: `1px solid ${C.border}`, paddingTop: 16, display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 6 }}>
          <span style={{ fontFamily: FONT, fontSize: 12, color: C.textFaint }}>tDCBAM · Signature Verification · Test Protocol Tool</span>
          <span style={{ fontFamily: FONT, fontSize: 12, color: C.textFaint }}>Matches proposed notebook evaluation exactly · K=1 Episodic Protocol</span>
        </div>
      </div>
    </div>
  );
}