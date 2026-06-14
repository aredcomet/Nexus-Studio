import re
import sys

def main():
    with open('src/App.svelte', 'r') as f:
        content = f.read()

    # Map of substring in SVG -> Material symbol
    replacements = [
        # Chat input stop & send
        (r'<svg class="w-3.5 h-3.5 fill-current" viewBox="0 0 24 24">\s*<rect.*?></rect>\s*</svg>', '<span class="material-symbols-outlined text-[14px]">stop</span>'),
        (r'<svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">\s*<path.*?M6 12L3.*?/>\s*</svg>', '<span class="material-symbols-outlined text-[16px]">send</span>'),
        
        # User / Assistant Avatars
        (r'<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">\s*<path.*?M15\.75 6a3\.75.*?/>\s*</svg>', '<span class="material-symbols-outlined text-[14px]">person</span>'),
        (r'<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">\s*<path.*?M3\.75 13\.5l10\.5.*?/>\s*</svg>', '<span class="material-symbols-outlined text-[14px]">smart_toy</span>'),

        # Empty Chat Hero Icon
        (r'<svg class="w-8 h-8 text-\[var\(--text-muted\)\]" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">\s*<path.*?M9\.813 15\.904L9 21.*?/>\s*</svg>', '<span class="material-symbols-outlined text-[32px] text-[var(--text-muted)]">dashboard_customize</span>'),

        # Theme toggle (Sun / Moon)
        (r'<svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">\s*<path.*?M21\.752 15\.002A9\.718.*?/>\s*</svg>', '<span class="material-symbols-outlined text-[16px]">dark_mode</span>'),
        (r'<svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">\s*<path.*?M12 3v2\.25m0.*?/>\s*</svg>', '<span class="material-symbols-outlined text-[16px]">light_mode</span>'),

        # Parameters / Sidebar / Search
        (r'<svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">\s*<path.*?M9\.594 3\.94c.*?/>\s*<path.*?M15 12a3.*?/>\s*</svg>', '<span class="material-symbols-outlined text-[16px]">tune</span>'),
        (r'<svg class="w-3\.5 h-3\.5 text-\[var\(--text-muted\)\]" fill="none" stroke="currentColor" stroke-width="1\.5" viewBox="0 0 24 24">\s*<path.*?M12 18a3\.75.*?/>\s*</svg>', '<span class="material-symbols-outlined text-[14px] text-[var(--text-muted)]">psychology</span>'),
        
        # Tool execution/result
        (r'<svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">\s*<path.*?M11\.42 15\.17L17\.25.*?/>\s*</svg>', '<span class="material-symbols-outlined text-[16px]">build</span>'),
        
        # Close / Info
        (r'<svg class="w-3\.5 h-3\.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M10\.5 6h9\.75M10\.5.*?/></svg>', '<span class="material-symbols-outlined text-[14px]">format_list_bulleted</span>'),
        
        # Caret down / Chevron down
        (r'<svg class="w-3 h-3 text-\[var\(--text-muted\)\]" fill="none" stroke="currentColor" stroke-width="1\.5" viewBox="0 0 24 24">\s*<path stroke-linecap="round" stroke-linejoin="round" d="M19\.5 8\.25l-7\.5 7\.5-7\.5-7\.5" />\s*</svg>', '<span class="material-symbols-outlined text-[12px] text-[var(--text-muted)]">expand_more</span>'),
        (r'<svg class="w-3 h-3 transition-transform duration-200 \{isExpanded \? \'rotate-180\' : \'\'\}" fill="none" stroke="currentColor" stroke-width="1\.5" viewBox="0 0 24 24">\s*<path stroke-linecap="round" stroke-linejoin="round" d="M19\.5 8\.25l-7\.5 7\.5-7\.5-7\.5" />\s*</svg>', '<span class="material-symbols-outlined text-[12px] transition-transform duration-200 {isExpanded ? \'rotate-180\' : \'\'}">expand_more</span>'),
        
        # Accordion chevron
        (r'<svg class="w-3 h-3 mt-1 text-\[var\(--text-muted\)\] transition-transform \{expandedToolArgs\[key\] \? \'rotate-90\' : \'\'\}" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg>', '<span class="material-symbols-outlined text-[12px] mt-1 text-[var(--text-muted)] transition-transform {expandedToolArgs[key] ? \'rotate-90\' : \'\'}">chevron_right</span>'),
        (r'<svg class="w-3 h-3 mt-1 text-\[var\(--text-muted\)\] transition-transform \{expandedToolResults\[key\] \? \'rotate-90\' : \'\'\}" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg>', '<span class="material-symbols-outlined text-[12px] mt-1 text-[var(--text-muted)] transition-transform {expandedToolResults[key] ? \'rotate-90\' : \'\'}">chevron_right</span>'),

    ]

    for pattern, repl in replacements:
        content = re.sub(pattern, repl, content)

    with open('src/App.svelte', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    main()
