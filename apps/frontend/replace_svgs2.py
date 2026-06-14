import re
import sys

def main():
    with open('src/App.svelte', 'r') as f:
        content = f.read()

    replacements = [
        # Line 1325: Toast error (X icon) - usually M6 18L18 6M6 6l12 12
        (r'<svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">\s*<path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />\s*</svg>', '<span class="material-symbols-outlined text-[16px]">close</span>'),
        
        # Line 1448: Warning icon - M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z
        (r'<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">\s*<path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h\.01m-6\.938 4h13\.856c1\.54 0 2\.502-1\.667 1\.732-3L13\.732 4c-\.77-1\.333-2\.694-1\.333-3\.464 0L3\.34 16c-\.77 1\.333\.192 3 1\.732 3z" />\s*</svg>', '<span class="material-symbols-outlined text-[14px]">warning</span>'),

        # Line 1461: Modal icon (Server/System) - M2.25 12.75V12A2.25...
        (r'<svg class="w-3.5 h-3.5 text-\[var\(--text-muted\)\]" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M2\.25 12\.75V12A2\.25 2\.25 0 014\.5 9\.75h15a2\.25 2\.25 0 012\.25 2\.25v\.75m-19\.5 0A2\.25 2\.25 0 004\.5 15h15a2\.25 2\.25 0 002\.25-2\.25m-19\.5 0v\.158c0 \.882\.365 1\.722 1 2\.302l1\.62 1\.458a2\.25 2\.25 0 001\.5 1\.571h8\.22a2\.25 2\.25 0 001\.5-\.57l1\.62-1\.459a2\.25 2\.25 0 00\.6-2\.302V12\.75m-19\.5 0V7\.5A2\.25 2\.25 0 014\.5 5\.25h5\.053c\.488 0 \.954\.19 1\.302\.53l1\.378 1\.378c\.348\.348\.814\.538 1\.302\.538h5\.217A2\.25 2\.25 0 0121 9\.964V12\.75" /></svg>', '<span class="material-symbols-outlined text-[14px] text-[var(--text-muted)]">dns</span>'),

        # Line 1532: Trash in modal
        (r'<svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">\s*<path stroke-linecap="round" stroke-linejoin="round" d="M14\.74 9l-\.346 9m-4\.788 0L9\.26 9m9\.968-3\.21c\.342\.052\.682\.107 1\.022\.166m-1\.022-\.165L18\.16 19\.673a2\.25 2\.25 0 01-2\.244 2\.077H8\.084a2\.25 2\.25 0 01-2\.244-2\.077L4\.772 5\.79m14\.456 0a48\.108 48\.108 0 00-3\.478-\.397m-12 \.562c\.34-\.059\.68-\.114 1\.022-\.165m0 0a48\.11 48\.11 0 013\.478-\.397m7\.5 0v-\.916c0-1\.18-\.91-2\.164-2\.09-2\.201a51\.964 51\.964 0 00-3\.32 0c-1\.18\.037-2\.09 1\.022-2\.09 2\.201v\.916m7\.5 0a48\.667 48\.667 0 00-7\.5 0" />\s*</svg>', '<span class="material-symbols-outlined text-[16px]">delete</span>'),

        # Line 1551: Close in modal
        (r'<svg class="w-3 h-3 text-\[var\(--text-muted\)\]" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">\s*<path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />\s*</svg>', '<span class="material-symbols-outlined text-[12px] text-[var(--text-muted)]">close</span>'),

        # Line 1894 & 1923: Empty chat options icons (Server & Memory/Chip)
        (r'<svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M3\.75 6A2\.25 2\.25 0 016 3\.75h2\.25A2\.25 2\.25 0 0110\.5 6v2\.25a2\.25 2\.25 0 01-2\.25 2\.25H6a2\.25 2\.25 0 01-2\.25-2\.25V6zM3\.75 15\.75A2\.25 2\.25 0 016 13\.5h2\.25a2\.25 2\.25 0 012\.25 2\.25V18a2\.25 2\.25 0 01-2\.25 2\.25H6A2\.25 2\.25 0 013\.75 18v-2\.25zM13\.5 6a2\.25 2\.25 0 012\.25-2\.25H18A2\.25 2\.25 0 0120\.25 6v2\.25A2\.25 2\.25 0 0118 10\.5h-2\.25a2\.25 2\.25 0 01-2\.25-2\.25V6zM13\.5 15\.75a2\.25 2\.25 0 012\.25-2\.25H18a2\.25 2\.25 0 012\.25 2\.25V18A2\.25 2\.25 0 0118 20\.25h-2\.25A2\.25 2\.25 0 0113\.5 18v-2\.25z" /></svg>', '<span class="material-symbols-outlined text-[12px]">grid_view</span>'),
        (r'<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M8\.25 3v1\.5M4\.5 8\.25H3m18 0h-1\.5M4\.5 12H3m18 0h-1\.5m-15 3\.75H3m18 0h-1\.5M8\.25 19\.5V21M12 3v1\.5m0 15V21m3\.75-18v1\.5m0 15V21m-9-1\.5h10\.5a2\.25 2\.25 0 002\.25-2\.25V6\.75a2\.25 2\.25 0 00-2\.25-2\.25H6\.75A2\.25 2\.25 0 004\.5 6\.75v10\.5a2\.25 2\.25 0 002\.25 2\.25z" /></svg>', '<span class="material-symbols-outlined text-[14px]">memory</span>'),

        # Line 2143: MCP server chevron down
        (r'<svg class="w-3\.5 h-3\.5 transition-transform duration-150 \{mcpExpandedServers\[server\.name\] \? \'rotate-180\' : \'\'\}" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">\s*<path stroke-linecap="round" stroke-linejoin="round" d="M19\.5 8\.25l-7\.5 7\.5-7\.5-7\.5" />\s*</svg>', '<span class="material-symbols-outlined text-[14px] transition-transform duration-150 {mcpExpandedServers[server.name] ? \'rotate-180\' : \'\'}">expand_more</span>'),

        # Line 2266: Folder chevron right
        (r'<svg class="w-3 h-3 text-\[var\(--accent-color\)\] transition-transform duration-150 \{folder\.isExpanded \? \'rotate-90\' : \'\'\}" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">\s*<path stroke-linecap="round" stroke-linejoin="round" d="M8\.25 4\.5l7\.5 7\.5-7\.5 7\.5" />\s*</svg>', '<span class="material-symbols-outlined text-[12px] text-[var(--accent-color)] transition-transform duration-150 {folder.isExpanded ? \'rotate-90\' : \'\'}">chevron_right</span>'),

        # Line 2269: Folder open icon
        (r'<svg class="w-3\.5 h-3\.5 text-\[var\(--text-secondary\)\]" fill="none" stroke="currentColor" stroke-width="1\.5" viewBox="0 0 24 24">\s*<path stroke-linecap="round" stroke-linejoin="round" d="M2\.25 12\.75V12A2\.25 2\.25 0 014\.5 9\.75h15a2\.25 2\.25 0 012\.25 2\.25v\.75m-19\.5 0A2\.25 2\.25 0 004\.5 15h15a2\.25 2\.25 0 002\.25-2\.25m-19\.5 0v\.158c0 \.882\.365 1\.722 1 2\.302l1\.62 1\.458a2\.25 2\.25 0 001\.5 1\.571h8\.22a2\.25 2\.25 0 001\.5-\.57l1\.62-1\.459a2\.25 2\.25 0 00\.6-2\.302V12\.75m-19\.5 0V7\.5A2\.25 2\.25 0 014\.5 5\.25h5\.053c\.488 0 \.954\.19 1\.302\.53l1\.378 1\.378c\.348\.348\.814\.538 1\.302\.538h5\.217A2\.25 2\.25 0 0121 9\.964V12\.75" />\s*</svg>', '<span class="material-symbols-outlined text-[14px] text-[var(--text-secondary)]">folder</span>'),
        (r'<svg class="w-3\.5 h-3\.5 text-\[var\(--text-secondary\)\]" fill="none" stroke="currentColor" stroke-width="1\.5" viewBox="0 0 24 24">\s*<path stroke-linecap="round" stroke-linejoin="round" d="M2\.25 12\.75V12A2\.25 2\.25 0 014\.5 9\.75h15a2\.25 2\.25 0 012\.25 2\.25v\.75m-19\.5 0A2\.25 2\.25 0 004\.5 15h15a2\.25 2\.25 0 002\.25-2\.25m-19\.5 0v\.158c0 \.882\.365 1\.722 1 2\.302l1\.62 1\.458a2\.25 2\.25 0 001\.5 1\.571h8\.22a2\.25 2\.25 0 001\.5-\.57l1\.62-1\.459a2\.25 2\.25 0 00\.6-2\.302V12\.75m-19\.5 0V7\.5A2\.25 2\.25 0 014\.5 5\.25h5\.053c\.488 0 \.954\.19 1\.302\.53l1\.378 1\.378c\.348\.348\.814\.538 1\.302\.538h5\.217A2\.25 2\.25 0 0121 9\.964V12\.75" />\s*</svg>', '<span class="material-symbols-outlined text-[14px] text-[var(--text-secondary)]">folder</span>'),
        # Wait, the folder icon was probably M3.75 9h16.5m-16.5 6.75h16.5... 
        # I'll replace any remaining SVGs globally if they match certain paths.
    ]

    for pattern, repl in replacements:
        content = re.sub(pattern, repl, content)

    # Some fallback generic replacers for remaining SVGs based on their line numbers
    # Line 2276, 2281, 2286, 2291 - these are chat icons
    content = re.sub(r'<svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">\s*<path stroke-linecap="round" stroke-linejoin="round" d="M12 20\.25c4\.97 0 9-3\.694 9-8\.25s-4\.03-8\.25-9-8\.25S3 7\.444 3 12c0 2\.104\.859 4\.023 2\.3 5\.487l\.828\.842-\.354 1\.614c-\.18\.82\.638 1\.494 1\.358 1\.128l1\.625-\.827c1\.214\.462 2\.535\.72 3\.923\.72z" />\s*</svg>', '<span class="material-symbols-outlined text-[12px]">chat_bubble</span>', content)
    content = re.sub(r'<svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">\s*<path stroke-linecap="round" stroke-linejoin="round" d="M2\.25 12\.75V12A2\.25 2\.25 0 014\.5 9\.75h15a2\.25 2\.25 0 012\.25 2\.25v\.75m-19\.5 0A2\.25 2\.25 0 004\.5 15h15a2\.25 2\.25 0 002\.25-2\.25m-19\.5 0v\.158c0 \.882\.365 1\.722 1 2\.302l1\.62 1\.458a2\.25 2\.25 0 001\.5 1\.571h8\.22a2\.25 2\.25 0 001\.5-\.57l1\.62-1\.459a2\.25 2\.25 0 00\.6-2\.302V12\.75m-19\.5 0V7\.5A2\.25 2\.25 0 014\.5 5\.25h5\.053c\.488 0 \.954\.19 1\.302\.53l1\.378 1\.378c\.348\.348\.814\.538 1\.302\.538h5\.217A2\.25 2\.25 0 0121 9\.964V12\.75" />\s*</svg>', '<span class="material-symbols-outlined text-[12px]">folder</span>', content)

    # Chat file / document
    content = re.sub(r'<svg class="w-3\.5 h-3\.5 text-\[var\(--text-muted\)\] group-hover:text-\[var\(--accent-color\)\] transition-colors" fill="none" stroke="currentColor" stroke-width="1\.5" viewBox="0 0 24 24">\s*<path stroke-linecap="round" stroke-linejoin="round" d="M19\.5 14\.25v-2\.625a3\.375 3\.375 0 00-3\.375-3\.375h-1\.5A1\.125 1\.125 0 0113\.5 7\.125v-1\.5a3\.375 3\.375 0 00-3\.375-3\.375H8\.25m3\.75 9v6m3-3H9m1\.5-12H5\.625c-\.621 0-1\.125\.504-1\.125 1\.125v17\.25c0 \.621\.504 1\.125 1\.125 1\.125h12\.75c\.621 0 1\.125-\.504 1\.125-1\.125V11\.25a9 9 0 00-9-9z" />\s*</svg>', '<span class="material-symbols-outlined text-[14px] text-[var(--text-muted)] group-hover:text-[var(--accent-color)] transition-colors">description</span>', content)
    
    # Edit / Branch / Delete inside chat items
    content = re.sub(r'<svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="1\.5" viewBox="0 0 24 24">\s*<path stroke-linecap="round" stroke-linejoin="round" d="M16\.862 4\.487l1\.687-1\.688a1\.875 1\.875 0 112\.652 2\.652L6\.832 19\.82a4\.5 4\.5 0 01-1\.897 1\.13l-2\.685\.8\.8-2\.685a4\.5 4\.5 0 011\.13-1\.897L16\.863 4\.487zm0 0L19\.5 7\.125" />\s*</svg>', '<span class="material-symbols-outlined text-[12px]">edit</span>', content)
    content = re.sub(r'<svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="1\.5" viewBox="0 0 24 24">\s*<path stroke-linecap="round" stroke-linejoin="round" d="M7\.5 19\.5h-1\.5A2\.25 2\.25 0 013\.75 17\.25v-10\.5A2\.25 2\.25 0 016 4\.5h1\.5m10\.5 15h1\.5a2\.25 2\.25 0 002\.25-2\.25v-10\.5A2\.25 2\.25 0 0019\.5 4\.5h-1\.5M12 9v10\.5M15 15l-3 3-3-3" />\s*</svg>', '<span class="material-symbols-outlined text-[12px]">fork_right</span>', content)
    content = re.sub(r'<svg class="w-3 h-3" fill="none" stroke="currentColor" stroke-width="1\.5" viewBox="0 0 24 24">\s*<path stroke-linecap="round" stroke-linejoin="round" d="M14\.74 9l-\.346 9m-4\.788 0L9\.26 9m9\.968-3\.21c\.342\.052\.682\.107 1\.022\.166m-1\.022-\.165L18\.16 19\.673a2\.25 2\.25 0 01-2\.244 2\.077H8\.084a2\.25 2\.25 0 01-2\.244-2\.077L4\.772 5\.79m14\.456 0a48\.108 48\.108 0 00-3\.478-\.397m-12 \.562c\.34-\.059\.68-\.114 1\.022-\.165m0 0a48\.11 48\.11 0 013\.478-\.397m7\.5 0v-\.916c0-1\.18-\.91-2\.164-2\.09-2\.201a51\.964 51\.964 0 00-3\.32 0c-1\.18\.037-2\.09 1\.022-2\.09 2\.201v\.916m7\.5 0a48\.667 48\.667 0 00-7\.5 0" />\s*</svg>', '<span class="material-symbols-outlined text-[12px]">delete</span>', content)


    with open('src/App.svelte', 'w') as f:
        f.write(content)

if __name__ == '__main__':
    main()
