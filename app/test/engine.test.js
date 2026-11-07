'use strict';
const test = require('node:test');
const assert = require('node:assert');
const { lineSplitter, resolvePython, COMMANDS, Engine } = require('../engine');

test('lineSplitter reassembles lines across chunk boundaries and CRLF', () => {
  const lines = [];
  const feed = lineSplitter((l) => lines.push(l));
  feed(Buffer.from('{"event":"lo'));
  feed(Buffer.from('g"}\r\nsecond\nthi'));
  feed(Buffer.from('rd\r\n\n'));
  assert.deepStrictEqual(lines, ['{"event":"log"}', 'second', 'third']);
});

test('lineSplitter keeps multi-byte characters intact when a chunk is a whole string', () => {
  const lines = [];
  lineSplitter((l) => lines.push(l))(Buffer.from('héllo 🦘\n', 'utf8'));
  assert.deepStrictEqual(lines, ['héllo 🦘']);
});

test('resolvePython prefers an explicit path', () => {
  assert.strictEqual(resolvePython('  C:/py/python.exe '), 'C:/py/python.exe');
});

test('only whitelisted commands can be started', () => {
  assert.deepStrictEqual([...COMMANDS].sort(), ['export', 'generate', 'info', 'train']);
  const engine = new Engine(() => {});
  assert.throws(() => engine.start('rm -rf', {}), /Unknown command/);
});

test('engine forwards protocol events, reports exit codes, and rejects concurrent jobs', async () => {
  const events = [];
  const engine = new Engine((e) => events.push(e));
  // `python -m kangaroo info` against the real engine would need torch; use a stub interpreter instead.
  const stub = process.execPath; // node, which will fail on `-m kangaroo` -> exits non-zero
  const first = engine.start('train', { model: 'x', dataset: 'y' }, stub);
  assert.throws(() => engine.start('train', {}, stub), /already running/);
  await first;
  assert.ok(events.some((e) => e.event === 'error'), 'a crashing engine must surface an error event');
  assert.strictEqual(engine.running, false);
});

test('engine reports a missing interpreter instead of hanging', async () => {
  const events = [];
  const engine = new Engine((e) => events.push(e));
  await engine.start('train', {}, 'definitely-not-a-real-python-binary');
  const err = events.find((e) => e.event === 'error');
  assert.ok(err && /Could not start Python/.test(err.message), JSON.stringify(events));
});
