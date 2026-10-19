'use strict';

/*
 * hexo-renderer-marked parses Markdown before hexo-filter-mathjax runs. A line
 * containing only "=" inside a display equation can therefore become a Setext
 * heading, and underscores can become emphasis. Temporarily replace display
 * equations with neutral tokens, then restore them before MathJax renders.
 */

const TOKEN_PREFIX = 'HEXOMATHBLOCKTOKEN';

function protectBlocks(content) {
  const lines = content.split(/\r?\n/);
  const blocks = [];
  let fence = null;

  for (let index = 0; index < lines.length; index++) {
    const fenceMatch = lines[index].match(/^[\t ]*(`{3,}|~{3,})/);
    if (fenceMatch) {
      const marker = fenceMatch[1];
      if (!fence) {
        fence = { character: marker[0], length: marker.length };
      } else if (marker[0] === fence.character && marker.length >= fence.length) {
        fence = null;
      }
      continue;
    }
    if (fence) continue;

    const opening = lines[index].match(/^([\t ]*(?:[-*+]\s+)?)\$\$[\t ]*$/);
    if (!opening) continue;

    let closing = index + 1;
    while (closing < lines.length && !/^[\t ]*\$\$[\t ]*$/.test(lines[closing])) {
      closing++;
    }
    if (closing === lines.length) continue;

    const token = `${TOKEN_PREFIX}${blocks.length}END`;
    blocks.push({ token, tex: lines.slice(index + 1, closing).join('\n') });
    lines.splice(index, closing - index + 1, `${opening[1]}${token}`);
  }

  return { blocks, content: lines.join('\n') };
}

hexo.extend.filter.register('before_post_render', data => {
  if (!data.mathjax) return data;

  const protectedContent = protectBlocks(data.content);
  data.content = protectedContent.content;
  data.__protectedMathBlocks = protectedContent.blocks;
  return data;
}, 0);

hexo.extend.filter.register('after_post_render', data => {
  const blocks = data.__protectedMathBlocks || [];
  for (const { token, tex } of blocks) {
    data.content = data.content.replace(token, `$$\n${tex}\n$$`);
  }
  delete data.__protectedMathBlocks;
  return data;
}, 0);
