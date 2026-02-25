
import { useState, useRef } from 'react';

export function useVerification() {
    const [dataset, setDataset] = useState('cedar');
    const [split, setSplit] = useState('70_15_15');
    const [supports, setSupports] = useState([null, null, null]);
    const [previews, setPreviews] = useState([null, null, null]);
    const [queryFile, setQueryFile] = useState(null);
    const [queryPreview, setQueryPreview] = useState(null);

    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const resultsRef = useRef(null);
    const allReady = !supports.includes(null) && !!queryFile;

    const handleSupport = (i) => (e) => {
        const f = e.target.files[0];
        if (!f) return;
        const ns = [...supports], np = [...previews];
        ns[i] = f;
        np[i] = URL.createObjectURL(f);
        setSupports(ns);
        setPreviews(np);
    };

    const clearSupport = (i) => () => {
        const ns = [...supports], np = [...previews];
        ns[i] = null;
        np[i] = null;
        setSupports(ns);
        setPreviews(np);
    };

    const handleQuery = (e) => {
        const f = e.target.files[0];
        if (!f) return;
        setQueryFile(f);
        setQueryPreview(URL.createObjectURL(f));
    };

    const clearQuery = () => {
        setQueryFile(null);
        setQueryPreview(null);
    };

    const verify = async () => {
        setLoading(true);
        setResult(null);
        setError(null);

        const fd = new FormData();
        fd.append('dataset', dataset);
        fd.append('split', split);
        fd.append('support_file_1', supports[0]);
        fd.append('support_file_2', supports[1]);
        fd.append('support_file_3', supports[2]);
        fd.append('query_file', queryFile);

        try {
            const r = await fetch('http://localhost:8000/verify', { method: 'POST', body: fd });
            if (!r.ok) {
                const d = await r.json();
                throw new Error(d.detail || 'Server error');
            }
            setResult(await r.json());
            setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 120);
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    return {
        dataset, setDataset, split, setSplit,
        supports, previews, handleSupport, clearSupport,
        queryFile, queryPreview, handleQuery, clearQuery,
        loading, result, error, resultsRef, allReady, verify
    };
}