'use strict';
const $ = id => document.getElementById(id);
const colors = [[242,60,60],[60,113,242],[166,242,60],[242,60,219],[60,242,211],[242,180,60]];
const packets = new Map();
let manifest, generation = 0;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
async function json(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status}: ${url}`);
  return response.json();
}
async function packet(url) {
  if (!packets.has(url)) packets.set(url, json(url));
  return packets.get(url);
}
function card(model) {
  const node = element('article');
  node.append(element('h2', model.label));
  const metrics = model.metrics;
  let text = `${metrics.images} images · ${metrics.mean_candidates.toFixed(2)} candidates/image`;
  if (model.has_gt) {
    text += ` · complete ${metrics.complete_gt}/${metrics.gt}`;
    text += ` · whole-image ${metrics.successful_images}/${metrics.images}`;
    for (const n of [3,5,10,50]) text += ` · TOP-${n} ${model.top_n_metrics[n].complete_gt}/${metrics.gt}`;
  } else text += ' · no GT: no recall is reported';
  node.append(element('p', text, 'detail'));
  return node;
}
async function draw(model, index, layer, n, request) {
  const base = `${model.path}/${String(index).padStart(5,'0')}`;
  const data = await packet(`${base}/candidates.json`);
  const image = new Image(); image.src = `${base}/image.png`; await image.decode();
  if (request !== generation) return;
  const node = element('article');
  node.dataset.model = model.label;
  node.append(element('h2', layer === 'gt' ? 'Ground truth' : model.label));
  const canvas = element('canvas'); canvas.width = data.width; canvas.height = data.height;
  const context = canvas.getContext('2d'); context.drawImage(image, 0, 0);
  let masks = [];
  if (layer === 'gt') masks = data.gt_masks || [];
  else {
    const order = data.scores.map((score,i) => i).sort((a,b) => data.scores[b]-data.scores[a] || a-b);
    const ids = layer === 'topn' ? order.slice(0,n) : order;
    masks = ids.map(i => data.masks[i]);
  }
  const pixels = context.getImageData(0,0,canvas.width,canvas.height);
  masks.forEach((rle,rank) => {
    let position = 0;
    rle.forEach((count,run) => {
      if (run % 2) for (let p = position; p < position + count; p++) {
        for (let c = 0; c < 3; c++) pixels.data[4*p+c] = Math.floor(pixels.data[4*p+c]*.55 + colors[rank%colors.length][c]*.45);
      }
      position += count;
    });
    if (position !== data.width*data.height) throw new Error('Invalid mask dimensions');
  });
  context.putImageData(pixels,0,0);
  node.append(canvas);
  const note = layer === 'gt' ? (data.gt_masks ? `${masks.length} annotated instances` : 'No GT annotations for this image') : `${masks.length}/${data.masks.length} actual candidates · score > ${data.score_threshold}`;
  node.append(element('p',note,'detail'));
  return node;
}
async function render() {
  const request = ++generation;
  const index = Number($('image').value), layer = $('layer').value;
  const n = Math.max(1,Math.min(200,Number($('topn').value) || 1)); $('topn').value = n;
  $('topn').disabled = layer !== 'topn';
  $('previous').disabled = index === 0; $('next').disabled = index === manifest.ids.length-1;
  const selected = manifest.models.filter((m,i) => $(`model-${i}`).checked);
  $('summary').replaceChildren(...selected.map(card)); $('gallery').replaceChildren();
  $('status').className = ''; $('status').textContent = 'Rendering native masks…';
  try {
    const shown = layer === 'gt' ? selected.slice(0,1) : selected;
    const cards = await Promise.all(shown.map(m => draw(m,index,layer,n,request)));
    if (request !== generation) return;
    $('gallery').replaceChildren(...cards.filter(Boolean));
    $('status').textContent = `${manifest.ids[index]} · ${layer === 'topn' ? `TOP-${n} by score` : layer} · ready`;
  } catch (error) {
    if (request !== generation) return;
    $('status').className = 'error'; $('status').textContent = error.message;
  }
}
async function main() {
  manifest = await json('manifest.json');
  if (!manifest.ids.length) throw new Error('No evaluation images');
  manifest.ids.forEach((id,i) => { const option = element('option',id); option.value = i; $('image').append(option); });
  manifest.models.forEach((model,i) => {
    const label = element('label'), input = element('input');
    input.type = 'checkbox'; input.id = `model-${i}`; input.checked = true; input.onchange = render;
    label.append(input,document.createTextNode(model.label)); $('models').append(label);
  });
  for (const id of ['image','layer','topn']) $(id).onchange = render;
  $('previous').onclick = () => { $('image').value = Number($('image').value)-1; render(); };
  $('next').onclick = () => { $('image').value = Number($('image').value)+1; render(); };
  await render();
}
main().catch(error => { $('status').className = 'error'; $('status').textContent = error.message; });
