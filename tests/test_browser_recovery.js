// Runs the production status functions against a small DOM double, without a server.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../ui/app.js'), 'utf8');
const stateFunctions = source.slice(source.indexOf('function setState('), source.indexOf('function sendCommand('));
const handler = source.slice(source.indexOf('function handle('), source.indexOf('/** Re-label already-rendered'));
function element() {
  return { dataset: {}, textContent: '', title: '', style: {}, classList: { toggle() {} } };
}
const els = Object.fromEntries(['dot', 'state', 'toggle', 'toggleLabel', 'captions',
  'meter', 'meterFill', 'latency'].map(key => [key, element()]));
const context = vm.createContext({ els, removed: 0 });
vm.runInContext(`
  let running = true;
  let inputState = null;
  let provisionalEl = { remove() { removed++; } };
  let currentId = 12;
  ${stateFunctions}
  ${handler}
`, context);
let checks = 0;
function check(condition, label) {
  assert.ok(condition, label);
  checks++;
  console.log(`PASS ${label}`);
}
function handle(message) {
  context.message = message;
  vm.runInContext('handle(message)', context);
}

handle({ type: 'input', state: 'recovering', wanted: true, running: false });
check(els.toggleLabel.textContent === 'Pause', 'recovery preserves the pause action');
check(context.removed === 0, 'device interruption preserves provisional captions for finalization');
check(els.state.textContent === 'reconnecting audio', 'recovery is distinct from a user pause');
handle({ type: 'status', state: 'recovering', wanted: true, running: false });
check(els.state.textContent === 'reconnecting audio', 'compatibility status cannot erase the recovery message');

handle({ type: 'input', state: 'failed', wanted: true, running: true,
  message: 'The previous input is still running.' });
handle({ type: 'status', state: 'listening', wanted: true, running: true });
check(els.state.textContent === 'The previous input is still running.', 'rollback remains explained while captions continue');
check(els.toggleLabel.textContent === 'Pause', 'failed switching leaves the previous capture controllable');
handle({ type: 'input', state: 'ready', wanted: true, running: true, active: { name: 'Test input' } });
check(els.state.textContent === 'Test input', 'ready state clears transient recovery text');

handle({ type: 'input', state: 'selected', wanted: false, running: false });
check(context.removed === 1 && els.toggleLabel.textContent === 'Start', 'a real pause discards provisional audio and offers start');
handle({ type: 'input', state: 'recovering', wanted: false, running: false });
check(els.state.textContent === 'checking input, capture stays paused', 'metadata probing does not imply recording or capture');
handle({ type: 'status', state: 'loading', model: 'new-model' });
check(els.state.textContent === 'loading new-model', 'a model restart clears old input recovery state');
console.log(`${checks} checks passed.`);
