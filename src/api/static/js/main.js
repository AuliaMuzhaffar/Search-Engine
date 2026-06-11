// DOM Elements
const queryInput = document.getElementById('queryInput');
const methodSelect = document.getElementById('methodSelect');
const resultsList = document.getElementById('resultsList');
const resultsStats = document.getElementById('resultsStats');
const resultsCount = document.getElementById('resultsCount');
const resultsLatency = document.getElementById('resultsLatency');
const resultsLoading = document.getElementById('resultsLoading');
const introSection = document.getElementById('introSection');
const searchContainer = document.getElementById('searchContainer');
const spellingSuggestion = document.getElementById('spellingSuggestion');
const spellingSuggestionLink = document.getElementById('spellingSuggestionLink');

// Modal Elements
const docModal = document.getElementById('docModal');
const modalDocType = document.getElementById('modalDocType');
const modalDocTitle = document.getElementById('modalDocTitle');
const modalDocContent = document.getElementById('modalDocContent');
const modalDocUrl = document.getElementById('modalDocUrl');

// Stats Modal Elements
const statsModal = document.getElementById('statsModal');
const statDocsCount = document.getElementById('statDocsCount');
const statTermsCount = document.getElementById('statTermsCount');
const statIndexBuilt = document.getElementById('statIndexBuilt');
const statRawDir = document.getElementById('statRawDir');
const statProcessedDir = document.getElementById('statProcessedDir');
const rebuildBtn = document.getElementById('rebuildBtn');

// Handle Search Event
async function handleSearch(event) {
    if (event) event.preventDefault();
    
    const query = queryInput.value.trim();
    const method = methodSelect.value;
    
    if (!query) return;

    // Adjust layout for search results view
    introSection.classList.add('scale-75', 'opacity-50', 'max-h-0', 'overflow-hidden', 'mb-0', 'pointer-events-none');
    searchContainer.classList.remove('py-12');
    searchContainer.classList.add('py-6');

    // Show Loading state
    resultsLoading.classList.remove('hidden');
    resultsList.innerHTML = '';
    resultsStats.classList.add('hidden');
    spellingSuggestion.classList.add('hidden');

    try {
        const response = await fetch(`/api/v1/search?q=${encodeURIComponent(query)}&method=${method}&top_k=15`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Hide Loading state
        resultsLoading.classList.add('hidden');
        
        // Display Stats
        resultsStats.classList.remove('hidden');
        resultsCount.innerText = data.total_results;
        resultsLatency.innerText = `${data.latency_ms} ms`;

        // Display spelling suggestion
        if (data.spelling_suggestion) {
            spellingSuggestionLink.innerText = data.spelling_suggestion;
            spellingSuggestionLink.onclick = (e) => {
                e.preventDefault();
                queryInput.value = data.spelling_suggestion;
                handleSearch();
            };
            spellingSuggestion.classList.remove('hidden');
        } else {
            spellingSuggestion.classList.add('hidden');
        }
        
        // Render Results
        if (data.results && data.results.length > 0) {
            resultsList.innerHTML = data.results.map(doc => `
                <div class="glass-panel p-6 rounded-2xl transition-all duration-300 hover:border-indigo-500/35 hover:-translate-y-1 shadow-lg hover:shadow-indigo-500/5 group">
                    <div class="flex items-start justify-between gap-4">
                        <div class="space-y-1">
                            <div class="flex items-center gap-2">
                                <span class="text-[10px] font-extrabold text-indigo-400 bg-indigo-950 px-2 py-0.5 rounded-md border border-indigo-900/30">
                                    #${doc.rank}
                                </span>
                                <h3 
                                    onclick="viewDocument('${doc.doc_id}')" 
                                    class="text-base font-semibold text-slate-100 hover:text-indigo-400 cursor-pointer transition-colors font-outfit"
                                >
                                    ${doc.title}
                                </h3>
                            </div>
                            <a 
                                href="${doc.url}" 
                                target="_blank" 
                                class="text-[10px] text-slate-500 hover:text-indigo-300 hover:underline transition-colors flex items-center gap-1"
                            >
                                ${doc.url}
                                <svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                                </svg>
                            </a>
                        </div>
                        
                        <!-- Relevance Badge -->
                        <div class="text-right">
                            <span class="inline-block text-xs font-bold font-outfit text-indigo-400 bg-indigo-500/10 border border-indigo-500/20 px-2.5 py-1 rounded-xl">
                                Score: ${doc.score}
                            </span>
                        </div>
                    </div>
                    
                    <!-- Snippet text -->
                    <p class="text-xs text-slate-400 font-light mt-3 leading-relaxed">
                        ${doc.snippet || 'Tidak ada pratinjau cuplikan teks.'}
                    </p>
                </div>
            `).join('');
        } else {
            resultsList.innerHTML = `
                <div class="glass-panel p-8 rounded-2xl text-center text-slate-400 space-y-2">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-slate-500 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <p class="text-sm font-medium">Tidak ada dokumen yang cocok dengan kata kunci pencarian.</p>
                </div>
            `;
        }
        
    } catch (err) {
        console.error(err);
        resultsLoading.classList.add('hidden');
        resultsList.innerHTML = `
            <div class="glass-panel p-8 rounded-2xl border-red-500/20 text-center text-red-400 space-y-2">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <p class="text-sm font-medium">Terjadi kesalahan koneksi API. Mohon coba beberapa saat lagi.</p>
            </div>
        `;
    }
}

// Open Document Detail modal
async function viewDocument(docId) {
    try {
        const response = await fetch(`/api/v1/documents/${encodeURIComponent(docId)}`);
        if (!response.ok) throw new Error('Failed to fetch document details');
        
        const doc = await response.json();
        
        modalDocTitle.innerText = doc.title;
        modalDocType.innerText = doc.type === 'raw' ? 'Original Text' : 'Stemmed Tokens';
        modalDocUrl.href = doc.url;
        
        // Format document content with linebreaks
        const paragraphs = doc.content.split('\n');
        modalDocContent.innerHTML = paragraphs.map(p => `<p class="mb-4">${p.trim()}</p>`).join('');
        
        // Show Modal
        docModal.classList.remove('hidden');
        docModal.classList.add('flex');
        document.body.classList.add('overflow-hidden');
    } catch (err) {
        alert(`Gagal memuat detail dokumen: ${err.message}`);
    }
}

// Close Document Detail modal
function closeDocModal() {
    docModal.classList.add('hidden');
    docModal.classList.remove('flex');
    document.body.classList.remove('overflow-hidden');
}

// Toggle Engine Stats Modal
async function toggleStatsModal() {
    if (statsModal.classList.contains('hidden')) {
        // Load stats first
        try {
            const response = await fetch('/api/v1/stats');
            if (response.ok) {
                const stats = await response.json();
                statDocsCount.innerText = stats.total_documents;
                statTermsCount.innerText = stats.total_indexed_terms;
                statIndexBuilt.innerText = stats.index_last_built;
                statRawDir.innerText = stats.raw_dir_exists ? 'Ready' : 'Not Found';
                statProcessedDir.innerText = stats.processed_dir_exists ? 'Ready' : 'Not Found';
            }
        } catch (err) {
            console.error('Failed to load stats:', err);
        }
        
        statsModal.classList.remove('hidden');
        document.body.classList.add('overflow-hidden');
    } else {
        statsModal.classList.add('hidden');
        document.body.classList.remove('overflow-hidden');
    }
}

// Trigger Rebuild Index in background
async function triggerRebuildIndex() {
    rebuildBtn.disabled = true;
    rebuildBtn.innerText = 'Rebuilding Index...';
    rebuildBtn.classList.remove('bg-indigo-600', 'hover:bg-indigo-500');
    rebuildBtn.classList.add('bg-slate-700');
    
    try {
        const response = await fetch('/api/v1/index/rebuild', { method: 'POST' });
        if (response.ok) {
            alert('Proses pembuatan indeks baru telah dimulai di latar belakang.');
            setTimeout(() => {
                rebuildBtn.disabled = false;
                rebuildBtn.innerText = 'Rebuild Index';
                rebuildBtn.classList.add('bg-indigo-600', 'hover:bg-indigo-500');
                rebuildBtn.classList.remove('bg-slate-700');
                toggleStatsModal(); // Refresh modal
            }, 3000);
        } else {
            throw new Error('Failed to trigger rebuild');
        }
    } catch (err) {
        alert(`Error triggering index rebuild: ${err.message}`);
        rebuildBtn.disabled = false;
        rebuildBtn.innerText = 'Rebuild Index';
        rebuildBtn.classList.add('bg-indigo-600', 'hover:bg-indigo-500');
        rebuildBtn.classList.remove('bg-slate-700');
    }
}

// Close modals when clicking outside content area
window.onclick = function(event) {
    if (event.target === docModal) {
        closeDocModal();
    } else if (event.target === statsModal) {
        toggleStatsModal();
    }
}
