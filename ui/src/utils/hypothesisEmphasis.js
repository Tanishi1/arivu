export function getEmphasisStyle(status) {
  switch (status) {
    case 'selected':
    case 'best':
    case 'validated':
      return { stroke: 'var(--thesis-color)', opacity: 1, dash: 'none' };
    case 'faint':
    case 'non-selected':
    case 'hypothesis':
      return { stroke: 'var(--border-active)', opacity: 0.5, dash: 'none' };
    case 'escape':
      return { stroke: 'var(--escape-valve-color)', opacity: 1, dash: '4 4' };
    case 'breached':
      return { stroke: 'var(--error)', opacity: 1, dash: 'none' };
    case 'trusted':
      return { stroke: 'var(--trust-color)', opacity: 1, dash: 'none' };
    default:
      return { stroke: 'var(--border-soft)', opacity: 0.3, dash: 'none' };
  }
}
