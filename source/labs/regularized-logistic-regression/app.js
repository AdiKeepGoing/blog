(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const lambdaOptions = [0, 0.01, 0.1, 1, 10, 100];
  const polynomialNames = degree => {
    const names = [];
    for (let total = 1; total <= degree; total++) {
      for (let p1 = total; p1 >= 0; p1--) {
        const p2 = total - p1;
        const left = p1 ? `x₁${p1 > 1 ? superscript(p1) : ""}` : "";
        const right = p2 ? `x₂${p2 > 1 ? superscript(p2) : ""}` : "";
        names.push(left + right);
      }
    }
    return names;
  };
  const featureNames = {
    linear: ["b", "x₁", "x₂"],
    quadratic: ["b", ...polynomialNames(2)],
    degree6: ["b", ...polynomialNames(6)]
  };

  const state = {
    optimizer: "batch",
    model: "quadratic",
    lambdaIndex: 2,
    learningRate: 0.18,
    targetEpoch: 60,
    speed: 1,
    sampleCount: 160,
    noise: 0.24,
    labelFlipRate: 0,
    earlyStopping: true,
    seed: 42,
    running: false,
    weights: [],
    data: [],
    train: [],
    test: [],
    stats: null,
    featureStats: null,
    order: [],
    cursor: 0,
    updates: 0,
    samplesSeen: 0,
    lastGradient: [],
    lossHistory: [],
    nextRecordEpoch: 0.2,
    bestTrainLoss: Infinity,
    lastImprovementEpoch: 0,
    stopReason: "",
    lastTick: 0
  };

  const els = {
    boundary: $("#boundaryCanvas"), loss: $("#lossCanvas"), run: $("#runButton"), step: $("#stepButton"), reset: $("#resetButton"),
    trainLoss: $("#trainLoss"), testLoss: $("#testLoss"), lossGap: $("#lossGap"), trainAcc: $("#trainAcc"), testAcc: $("#testAcc"), weightNorm: $("#weightNorm"),
    epoch: $("#epochLabel"), gradNorm: $("#gradNorm"), bars: $("#gradientBars"), iteration: $("#iterationLabel"),
    runState: $("#runState"), pulse: $("#pulse"), plotTitle: $("#plotTitle"), plotHint: $("#plotHint"), truth: $("#truthToggle"),
    lambda: $("#lambdaSlider"), lambdaValue: $("#lambdaValue"), lr: $("#lrSlider"), lrValue: $("#lrValue"),
    targetEpochValue: $("#targetEpochValue"), progressFill: $("#progressFill"),
    sample: $("#sampleSlider"), sampleValue: $("#sampleValue"), noise: $("#noiseSlider"), noiseValue: $("#noiseValue"),
    flip: $("#flipSlider"), flipValue: $("#flipValue"), earlyStop: $("#earlyStopToggle"),
    seed: $("#seedInput"), splitCounts: $("#splitCounts"), diagnosis: $("#diagnosis"), diagnosisTitle: $("#diagnosisTitle"), diagnosisText: $("#diagnosisText")
  };

  function superscript(number) {
    return String(number).replace(/[0-9]/g, digit => "⁰¹²³⁴⁵⁶⁷⁸⁹"[Number(digit)]);
  }

  // 固定种子的伪随机数生成器：相同 seed 一定得到相同数据。
  function mulberry32(seed) {
    return function() {
      let t = seed += 0x6D2B79F5;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function normal(rng) {
    const u = Math.max(rng(), 1e-12);
    const v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function shuffle(values, rng) {
    for (let i = values.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [values[i], values[j]] = [values[j], values[i]];
    }
    return values;
  }

  function sigmoid(z) {
    if (z >= 0) return 1 / (1 + Math.exp(-Math.min(z, 40)));
    const e = Math.exp(Math.max(z, -40));
    return e / (1 + e);
  }

  function trueBoundary(x1) { return 0.42 * x1 * x1 - 0.08 * x1 - 0.65; }

  // 数据生成流程：截断高斯特征 -> Sigmoid 概率 -> 伯努利采样标签。
  function generateData() {
    const rng = mulberry32(state.seed >>> 0);
    const points = [];
    for (let i = 0; i < state.sampleCount; i++) {
      const x1 = Math.max(-2.65, Math.min(2.65, normal(rng) * 1.16));
      const x2 = Math.max(-2.35, Math.min(2.35, normal(rng) * 1.02));
      const logit = (x2 - trueBoundary(x1)) / state.noise;
      let y = rng() < sigmoid(logit) ? 1 : 0;
      // 独立标签翻转模拟现实中的标注错误或测量错误。
      if (rng() < state.labelFlipRate) y = 1 - y;
      points.push({ x1, x2, y, split: "train" });
    }

    // 按类别分别抽取 25% 测试样本，避免某一类在测试集中过多或过少。
    const class0 = shuffle(points.filter(p => p.y === 0), rng);
    const class1 = shuffle(points.filter(p => p.y === 1), rng);
    [class0, class1].forEach(group => {
      const testN = Math.max(1, Math.round(group.length * 0.25));
      group.slice(0, testN).forEach(p => p.split = "test");
    });
    state.data = shuffle(points, rng);
    state.train = state.data.filter(p => p.split === "train");
    state.test = state.data.filter(p => p.split === "test");

    // 只用训练集计算均值和标准差，防止测试集信息泄漏。
    const mean1 = state.train.reduce((s, p) => s + p.x1, 0) / state.train.length;
    const mean2 = state.train.reduce((s, p) => s + p.x2, 0) / state.train.length;
    const sd1 = Math.sqrt(state.train.reduce((s, p) => s + (p.x1 - mean1) ** 2, 0) / state.train.length) || 1;
    const sd2 = Math.sqrt(state.train.reduce((s, p) => s + (p.x2 - mean2) ** 2, 0) / state.train.length) || 1;
    state.stats = { mean1, mean2, sd1, sd2 };
    els.splitCounts.textContent = `${state.train.length} train / ${state.test.length} test`;
    resetModel();
  }

  function rawFeatures(point, model = state.model) {
    const a = (point.x1 - state.stats.mean1) / state.stats.sd1;
    const b = (point.x2 - state.stats.mean2) / state.stats.sd2;
    const degree = model === "linear" ? 1 : model === "quadratic" ? 2 : 6;
    const values = [];
    for (let total = 1; total <= degree; total++) {
      for (let p1 = total; p1 >= 0; p1--) values.push((a ** p1) * (b ** (total - p1)));
    }
    return values;
  }

  // 多项式展开后再次只用训练集标准化每一列，避免高次项量纲失控。
  function fitFeatureStats() {
    const rows = state.train.map(point => rawFeatures(point));
    const count = rows[0]?.length || 0;
    const means = Array.from({ length: count }, (_, j) => rows.reduce((sum, row) => sum + row[j], 0) / rows.length);
    const sds = means.map((mean, j) => Math.sqrt(rows.reduce((sum, row) => sum + (row[j] - mean) ** 2, 0) / rows.length) || 1);
    state.featureStats = { means, sds };
  }

  function features(point) {
    const raw = rawFeatures(point);
    return [1, ...raw.map((value, j) => (value - state.featureStats.means[j]) / state.featureStats.sds[j])];
  }

  function resetModel() {
    state.running = false;
    fitFeatureStats();
    state.weights = new Array(featureNames[state.model].length).fill(0);
    state.updates = 0;
    state.samplesSeen = 0;
    state.cursor = 0;
    state.lastGradient = new Array(state.weights.length).fill(0);
    state.lossHistory = [];
    state.nextRecordEpoch = 0.2;
    state.bestTrainLoss = Infinity;
    state.lastImprovementEpoch = 0;
    state.stopReason = "";
    state.order = shuffledTrainOrder();
    setRunUi();
    recordMetrics(false);
    renderAll();
  }

  function shuffledTrainOrder() {
    const rng = mulberry32((state.seed * 7919 + state.updates * 104729 + 17) >>> 0);
    return shuffle(state.train.map((_, i) => i), rng);
  }

  function nextBatch() {
    if (state.optimizer === "batch") return state.train;
    const size = state.optimizer === "mini" ? Math.min(16, state.train.length) : 1;
    const batch = [];
    while (batch.length < size) {
      if (state.cursor >= state.order.length) {
        state.order = shuffledTrainOrder();
        state.cursor = 0;
      }
      batch.push(state.train[state.order[state.cursor++]]);
    }
    return batch;
  }

  // 一次真正的梯度更新。三种策略的区别只在 batch 的大小。
  function oneUpdate() {
    const batch = nextBatch();
    const grad = new Array(state.weights.length).fill(0);
    for (const point of batch) {
      const x = features(point);
      const p = sigmoid(dot(state.weights, x));
      for (let j = 0; j < grad.length; j++) grad[j] += (p - point.y) * x[j] / batch.length;
    }
    const lambda = lambdaOptions[state.lambdaIndex];
    // L2 正则项只作用于特征权重 w，不作用于偏置 b（索引 0）。
    for (let j = 1; j < grad.length; j++) grad[j] += (lambda / state.train.length) * state.weights[j];
    for (let j = 0; j < state.weights.length; j++) state.weights[j] -= state.learningRate * grad[j];
    state.lastGradient = grad;
    state.updates++;
    state.samplesSeen += batch.length;

    const epoch = currentEpoch();
    if (!state.weights.every(Number.isFinite)) finishTraining("diverged");

    // 每约 0.2 epoch 记录一次；批量梯度每步就是 1 epoch。
    if (!state.stopReason && (epoch >= state.nextRecordEpoch || epoch >= state.targetEpoch)) {
      recordMetrics(true);
      state.nextRecordEpoch = epoch + 0.2;
    }
    if (!state.stopReason && epoch >= state.targetEpoch) finishTraining("target");
  }

  function dot(a, b) { return a.reduce((s, v, i) => s + v * b[i], 0); }

  function evaluate(dataset, includeRegularization) {
    let loss = 0, correct = 0;
    dataset.forEach(point => {
      const p = sigmoid(dot(state.weights, features(point)));
      loss += -(point.y * Math.log(Math.max(p, 1e-12)) + (1 - point.y) * Math.log(Math.max(1 - p, 1e-12)));
      correct += (p >= 0.5 ? 1 : 0) === point.y ? 1 : 0;
    });
    loss /= Math.max(1, dataset.length);
    if (includeRegularization) {
      const penalty = state.weights.slice(1).reduce((s, w) => s + w * w, 0);
      loss += lambdaOptions[state.lambdaIndex] * penalty / (2 * state.train.length);
    }
    return { loss, accuracy: correct / Math.max(1, dataset.length) };
  }

  function currentEpoch() {
    return state.train.length ? state.samplesSeen / state.train.length : 0;
  }

  // 完整训练集/测试集指标只用于观察；测试集从不进入梯度计算。
  function recordMetrics(allowEarlyStop = true) {
    const train = evaluate(state.train, false);
    const objective = evaluate(state.train, true);
    const test = evaluate(state.test, false);
    const epoch = currentEpoch();
    state.lossHistory.push({ epoch, train: train.loss, test: test.loss });
    if (state.lossHistory.length > 600) state.lossHistory.shift();
    els.trainLoss.textContent = formatLoss(train.loss);
    els.testLoss.textContent = formatLoss(test.loss);
    els.lossGap.textContent = `${test.loss - train.loss >= 0 ? "+" : ""}${(test.loss - train.loss).toFixed(3)}`;
    els.trainAcc.textContent = formatPct(train.accuracy);
    els.testAcc.textContent = formatPct(test.accuracy);
    els.weightNorm.textContent = weightNorm().toFixed(3);
    updateDiagnosis(train, test, epoch);

    // 早停：至少训练 10 epoch，且连续 10 epoch 没有超过 1e-4 的改进。
    if (objective.loss < state.bestTrainLoss - 1e-4) {
      state.bestTrainLoss = objective.loss;
      state.lastImprovementEpoch = epoch;
    } else if (allowEarlyStop && state.earlyStopping && epoch >= 10 && epoch - state.lastImprovementEpoch >= 10) {
      finishTraining("converged");
    }
  }

  function weightNorm() {
    return Math.sqrt(state.weights.slice(1).reduce((sum, weight) => sum + weight * weight, 0));
  }

  // 这只是学习提示，不是统计检验；真正判断要结合多次数据划分或交叉验证。
  function updateDiagnosis(train, test, epoch) {
    const lossGap = test.loss - train.loss;
    const accuracyGap = train.accuracy - test.accuracy;
    const finiteHistory = state.lossHistory.filter(item => Number.isFinite(item.test) && Number.isFinite(item.train));
    const minTest = Math.min(...finiteHistory.map(item => item.test));
    const bestIndex = finiteHistory.findIndex(item => item.test === minTest);
    const bestPoint = finiteHistory[bestIndex] || finiteHistory[0];
    const latest = finiteHistory.at(-1);
    const divergingTrend = epoch >= 12 && latest && bestPoint && latest.test > minTest + 0.025 && latest.train < bestPoint.train - 0.015;
    const strongGap = lossGap > 0.10 || accuracyGap > 0.08;
    const mildGap = lossGap > 0.05 || accuracyGap > 0.04;

    let level = "neutral", title = "还没有足够证据", text = "先训练几轮，再同时观察两条损失曲线、准确率差距和权重范数。";
    if (divergingTrend && strongGap) {
      level = "overfit";
      title = "出现明显过拟合信号";
      text = accuracyGap < .04
        ? `测试损失已明显回升，但准确率只差 ${(accuracyGap * 100).toFixed(1)} 个百分点：准确率可能掩盖“错误置信度变强”，所以要同时看损失。`
        : `测试损失已从最低点回升，同时训练损失仍下降；准确率差距为 ${(accuracyGap * 100).toFixed(1)} 个百分点。`;
    } else if (divergingTrend || strongGap) {
      level = "watch";
      title = "正在出现过拟合信号";
      text = divergingTrend ? "训练与测试损失开始分叉。继续训练，观察差距是否持续扩大。" : `测试损失比训练损失高 ${lossGap.toFixed(3)}，建议与较大 λ 做同数据对照。`;
    } else if (epoch >= 8 && !mildGap) {
      level = "good";
      title = "暂未看到明显过拟合";
      text = "训练与测试指标仍较接近。请记住：这不等于模型一定合适，还要与欠拟合情况比较。";
    }
    els.diagnosis.dataset.level = level;
    els.diagnosisTitle.textContent = title;
    els.diagnosisText.textContent = text;
  }

  function finishTraining(reason) {
    state.running = false;
    state.stopReason = reason;
    setRunUi();
  }

  function formatLoss(value) { return Number.isFinite(value) ? value.toFixed(3) : "发散"; }
  function formatPct(value) { return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—"; }

  function setupCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr;
      canvas.height = height * dpr;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height };
  }

  function drawBoundaryPlot() {
    const { ctx, width, height } = setupCanvas(els.boundary);
    const pad = { l: 45, r: 20, t: 20, b: 36 };
    const xMin = -3, xMax = 3, yMin = -2.6, yMax = 2.6;
    const px = x => pad.l + (x - xMin) / (xMax - xMin) * (width - pad.l - pad.r);
    const py = y => height - pad.b - (y - yMin) / (yMax - yMin) * (height - pad.t - pad.b);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#f8f7f2";
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "#dedad1";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#788184";
    ctx.font = "9px ui-monospace, monospace";
    [-2, -1, 0, 1, 2].forEach(x => {
      ctx.beginPath(); ctx.moveTo(px(x), pad.t); ctx.lineTo(px(x), height - pad.b); ctx.stroke();
      ctx.fillText(x.toFixed(0), px(x) - 3, height - 17);
    });
    [-2, -1, 0, 1, 2].forEach(y => {
      ctx.beginPath(); ctx.moveTo(pad.l, py(y)); ctx.lineTo(width - pad.r, py(y)); ctx.stroke();
      ctx.fillText(y.toFixed(0), 18, py(y) + 3);
    });

    if (state.updates > 0) drawProbabilityField(ctx, px, py, xMin, xMax, yMin, yMax);

    if (els.truth.checked) {
      ctx.save();
      ctx.setLineDash([6, 5]);
      ctx.strokeStyle = "rgba(81,88,88,.8)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      let started = false;
      for (let x = xMin; x <= xMax; x += 0.035) {
        const y = trueBoundary(x);
        if (y >= yMin && y <= yMax) { started ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y)); started = true; }
      }
      ctx.stroke();
      ctx.restore();
    }

    if (state.updates > 0) drawModelContour(ctx, px, py, xMin, xMax, yMin, yMax);

    const ordered = [...state.data].sort((a, b) => (a.split === "test") - (b.split === "test"));
    ordered.forEach(point => {
      const x = px(point.x1), y = py(point.x2);
      ctx.beginPath(); ctx.arc(x, y, point.split === "test" ? 5.2 : 4.2, 0, Math.PI * 2);
      ctx.fillStyle = point.y ? "#f05a47" : "#1676b7";
      if (point.split === "test") {
        ctx.fillStyle = "#f8f7f2"; ctx.fill();
        ctx.lineWidth = 2.2; ctx.strokeStyle = point.y ? "#f05a47" : "#1676b7"; ctx.stroke();
        ctx.beginPath(); ctx.arc(x, y, 1.7, 0, Math.PI * 2); ctx.fillStyle = point.y ? "#f05a47" : "#1676b7"; ctx.fill();
      } else {
        ctx.fill(); ctx.lineWidth = 1.2; ctx.strokeStyle = "rgba(255,255,255,.8)"; ctx.stroke();
      }
    });
  }

  function probabilityAt(x1, x2) { return sigmoid(dot(state.weights, features({ x1, x2 }))); }

  function drawProbabilityField(ctx, px, py, xMin, xMax, yMin, yMax) {
    const cols = 42, rows = 36;
    const dx = (xMax - xMin) / cols, dy = (yMax - yMin) / rows;
    for (let i = 0; i < cols; i++) for (let j = 0; j < rows; j++) {
      const x = xMin + (i + .5) * dx, y = yMin + (j + .5) * dy;
      const p = probabilityAt(x, y);
      const alpha = 0.035 + Math.abs(p - .5) * .13;
      ctx.fillStyle = p >= .5 ? `rgba(240,90,71,${alpha})` : `rgba(22,118,183,${alpha})`;
      const left = px(x - dx / 2), right = px(x + dx / 2), top = py(y + dy / 2), bottom = py(y - dy / 2);
      ctx.fillRect(left, top, right - left + 1, bottom - top + 1);
    }
  }

  function drawModelContour(ctx, px, py, xMin, xMax, yMin, yMax) {
    const cols = 80, rows = 65;
    const dx = (xMax - xMin) / cols, dy = (yMax - yMin) / rows;
    ctx.save(); ctx.strokeStyle = "#15252b"; ctx.lineWidth = 3; ctx.lineCap = "round";
    for (let i = 0; i < cols; i++) for (let j = 0; j < rows; j++) {
      const x = xMin + i * dx, y = yMin + j * dy;
      const values = [
        probabilityAt(x, y) - .5,
        probabilityAt(x + dx, y) - .5,
        probabilityAt(x + dx, y + dy) - .5,
        probabilityAt(x, y + dy) - .5
      ];
      const crossings = [];
      const corners = [[x,y],[x+dx,y],[x+dx,y+dy],[x,y+dy]];
      [[0,1],[1,2],[2,3],[3,0]].forEach(([a,b]) => {
        if ((values[a] >= 0) !== (values[b] >= 0)) {
          const t = values[a] / (values[a] - values[b]);
          crossings.push([corners[a][0] + t * (corners[b][0] - corners[a][0]), corners[a][1] + t * (corners[b][1] - corners[a][1])]);
        }
      });
      if (crossings.length >= 2) {
        ctx.beginPath(); ctx.moveTo(px(crossings[0][0]), py(crossings[0][1])); ctx.lineTo(px(crossings[1][0]), py(crossings[1][1])); ctx.stroke();
        if (crossings.length === 4) { ctx.beginPath(); ctx.moveTo(px(crossings[2][0]), py(crossings[2][1])); ctx.lineTo(px(crossings[3][0]), py(crossings[3][1])); ctx.stroke(); }
      }
    }
    ctx.restore();
  }

  function drawLossChart() {
    const { ctx, width, height } = setupCanvas(els.loss);
    const pad = { l: 30, r: 8, t: 10, b: 20 };
    ctx.clearRect(0, 0, width, height); ctx.fillStyle = "#fbfaf6"; ctx.fillRect(0, 0, width, height);
    const history = state.lossHistory;
    if (!history.length) return;
    // 横轴固定为 0 -> 目标 epoch，训练越久也不会把已有轨迹挤到右侧。
    const maxEpoch = state.targetEpoch;
    const maxLoss = Math.min(2, Math.max(.75, ...history.flatMap(d => [d.train, d.test]).filter(Number.isFinite)));
    const minLoss = Math.max(0, Math.min(.25, ...history.flatMap(d => [d.train, d.test]).filter(Number.isFinite)) - .05);
    const x = e => pad.l + e / maxEpoch * (width - pad.l - pad.r);
    const y = l => height - pad.b - (l - minLoss) / Math.max(.01, maxLoss - minLoss) * (height - pad.t - pad.b);
    ctx.strokeStyle = "#e0ddd5"; ctx.lineWidth = 1;
    [0,.5,1].forEach(t => { const yy = pad.t + t * (height-pad.t-pad.b); ctx.beginPath(); ctx.moveTo(pad.l,yy); ctx.lineTo(width-pad.r,yy); ctx.stroke(); });
    const draw = (key, color, dash=[]) => {
      ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dash); ctx.beginPath();
      history.forEach((d, i) => { const xx=x(d.epoch), yy=y(Math.min(maxLoss,d[key])); i ? ctx.lineTo(xx,yy) : ctx.moveTo(xx,yy); }); ctx.stroke(); ctx.restore();
    };
    draw("train", "#f05a47"); draw("test", "#1676b7", [4,3]);
    ctx.font = "8px ui-monospace, monospace"; ctx.fillStyle = "#788184";
    ctx.fillText(maxLoss.toFixed(2), 3, 13); ctx.fillText(minLoss.toFixed(2), 3, height - pad.b);
    ctx.fillText("0", pad.l, height - 5); ctx.fillText(`${state.targetEpoch} epoch`, width - 55, height - 5);
  }

  function drawGradientBars() {
    const norm = Math.sqrt(state.lastGradient.reduce((s, g) => s + g * g, 0));
    els.gradNorm.textContent = `‖∇J‖ = ${norm.toFixed(3)}`;
    const max = Math.max(.001, ...state.lastGradient.map(Math.abs));
    const visible = featureNames[state.model].map((name, i) => ({ name, i, value: state.lastGradient[i] || 0 }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, state.model === "degree6" ? 8 : featureNames[state.model].length);
    els.bars.innerHTML = visible.map(({ name, value }) => {
      return `<div class="grad-row"><span>${name}</span><div class="grad-track"><i style="width:${Math.abs(value)/max*100}%;background:${value >= 0 ? "#2c7771" : "#f05a47"}"></i></div><b>${value.toFixed(3)}</b></div>`;
    }).join("");
  }

  function renderAll() {
    const epoch = currentEpoch();
    const shownEpoch = Math.min(epoch, state.targetEpoch);
    els.epoch.textContent = `${shownEpoch.toFixed(2)} / ${state.targetEpoch} epoch`;
    els.iteration.textContent = `${shownEpoch.toFixed(2)} / ${state.targetEpoch} epoch · 更新 ${state.updates} 次`;
    els.progressFill.style.width = `${Math.min(100, epoch / state.targetEpoch * 100)}%`;
    const modelTitle = { linear: "线性边界", quadratic: "二次特征", degree6: "六次多项式" }[state.model];
    els.plotTitle.textContent = `${modelTitle} · ${state.optimizer === "batch" ? "批量梯度" : state.optimizer === "mini" ? "小批量梯度" : "随机梯度"}`;
    els.plotHint.classList.toggle("hidden", state.updates > 0);
    drawBoundaryPlot(); drawLossChart(); drawGradientBars();
  }

  function setRunUi() {
    els.run.classList.toggle("running", state.running);
    els.run.querySelector("b").textContent = state.running ? "暂停训练" : state.stopReason ? "重新训练" : state.updates ? "继续训练" : "开始训练";
    els.pulse.classList.toggle("running", state.running);
    els.pulse.classList.toggle("complete", Boolean(state.stopReason));
    els.step.disabled = Boolean(state.stopReason);
    const labels = { target: "COMPLETE", converged: "CONVERGED", diverged: "STOPPED" };
    els.runState.textContent = state.running ? "RUNNING" : state.stopReason ? labels[state.stopReason] : state.updates ? "PAUSED" : "READY";
  }

  function animate(time) {
    if (state.running && time - state.lastTick > 38) {
      // 小批量和随机梯度每次看到的样本较少，因此一帧执行更多次更新。
      const baseWork = { batch: 1, mini: 2, sgd: 12 }[state.optimizer];
      const work = state.speed * baseWork;
      for (let i = 0; i < work && state.running; i++) oneUpdate();
      renderAll(); state.lastTick = time;
    }
    requestAnimationFrame(animate);
  }

  function selectButtons(container, value) {
    container.querySelectorAll("button").forEach(btn => btn.setAttribute("aria-pressed", String(btn.dataset.value === String(value))));
  }

  function bindControls() {
    $("#optimizerControl").addEventListener("click", e => {
      const button = e.target.closest("button"); if (!button) return;
      state.optimizer = button.dataset.value; selectButtons($("#optimizerControl"), state.optimizer); resetModel();
    });
    $("#modelControl").addEventListener("click", e => {
      const button = e.target.closest("button"); if (!button) return;
      state.model = button.dataset.value; selectButtons($("#modelControl"), state.model); resetModel();
    });
    $("#speedControl").addEventListener("click", e => {
      const button = e.target.closest("button"); if (!button) return;
      state.speed = Number(button.dataset.value); selectButtons($("#speedControl"), state.speed);
    });
    $("#epochControl").addEventListener("click", e => {
      const button = e.target.closest("button"); if (!button) return;
      state.targetEpoch = Number(button.dataset.value);
      els.targetEpochValue.textContent = state.targetEpoch;
      selectButtons($("#epochControl"), state.targetEpoch);
      resetModel();
    });
    els.lambda.addEventListener("input", () => {
      state.lambdaIndex = Number(els.lambda.value); els.lambdaValue.textContent = formatLambda(lambdaOptions[state.lambdaIndex]); setRangeFill(els.lambda); resetModel();
    });
    els.lr.addEventListener("input", () => { state.learningRate = Number(els.lr.value); els.lrValue.textContent = state.learningRate.toFixed(2); setRangeFill(els.lr); });
    els.run.addEventListener("click", () => {
      if (state.stopReason) resetModel();
      state.running = !state.running;
      setRunUi();
    });
    els.step.addEventListener("click", () => { state.running = false; oneUpdate(); setRunUi(); renderAll(); });
    els.reset.addEventListener("click", resetModel);
    els.truth.addEventListener("change", drawBoundaryPlot);
    els.sample.addEventListener("input", () => { els.sampleValue.textContent = els.sample.value; setRangeFill(els.sample); });
    els.noise.addEventListener("input", () => { els.noiseValue.textContent = Number(els.noise.value).toFixed(2); setRangeFill(els.noise); });
    els.flip.addEventListener("input", () => { els.flipValue.textContent = `${Math.round(Number(els.flip.value) * 100)}%`; setRangeFill(els.flip); });
    els.earlyStop.addEventListener("change", () => { state.earlyStopping = els.earlyStop.checked; });
    $("#regenerateButton").addEventListener("click", () => {
      state.sampleCount = Number(els.sample.value); state.noise = Number(els.noise.value); state.labelFlipRate = Number(els.flip.value); state.seed = Math.max(1, Number(els.seed.value) || 42); generateData();
    });
    $("#newSeedButton").addEventListener("click", () => { els.seed.value = state.seed = (Number(els.seed.value) || 42) + 1; state.sampleCount = Number(els.sample.value); state.noise = Number(els.noise.value); state.labelFlipRate = Number(els.flip.value); generateData(); });
    $$('[data-preset]').forEach(button => button.addEventListener("click", () => applyPreset(button.dataset.preset)));
    window.addEventListener("resize", renderAll);
  }

  function setRangeFill(input) {
    const min = Number(input.min), max = Number(input.max), value = Number(input.value);
    input.style.setProperty("--fill", `${(value - min) / (max - min) * 100}%`);
  }

  function formatLambda(value) {
    if (value === 0) return "0";
    if (value < 1) return value.toFixed(value < .1 ? 2 : 1);
    return value.toFixed(0);
  }

  function applyPreset(preset) {
    const config = {
      underfit: { model: "linear", lambda: 2, optimizer: "batch", lr: .18, epoch: 60, samples: 160, noise: .24, flip: 0, seed: 42, earlyStop: true },
      balanced: { model: "quadratic", lambda: 2, optimizer: "mini", lr: .14, epoch: 60, samples: 160, noise: .24, flip: 0, seed: 42, earlyStop: true },
      overregularized: { model: "quadratic", lambda: 4, optimizer: "batch", lr: .18, epoch: 60, samples: 160, noise: .24, flip: 0, seed: 42, earlyStop: true },
      overfit: { model: "degree6", lambda: 0, optimizer: "batch", lr: .12, epoch: 200, samples: 80, noise: .38, flip: .12, seed: 17, earlyStop: false, autoStart: true }
    }[preset];
    state.model = config.model; state.lambdaIndex = config.lambda; state.optimizer = config.optimizer; state.learningRate = config.lr; state.targetEpoch = config.epoch;
    state.sampleCount = config.samples; state.noise = config.noise; state.labelFlipRate = config.flip; state.seed = config.seed; state.earlyStopping = config.earlyStop;
    els.lambda.value = config.lambda; els.lr.value = config.lr; els.sample.value = config.samples; els.noise.value = config.noise; els.flip.value = config.flip; els.seed.value = config.seed; els.earlyStop.checked = config.earlyStop;
    els.targetEpochValue.textContent = config.epoch; els.lambdaValue.textContent = formatLambda(lambdaOptions[config.lambda]); els.lrValue.textContent = config.lr.toFixed(2); els.sampleValue.textContent = config.samples; els.noiseValue.textContent = config.noise.toFixed(2); els.flipValue.textContent = `${Math.round(config.flip * 100)}%`;
    selectButtons($("#modelControl"), state.model); selectButtons($("#optimizerControl"), state.optimizer); selectButtons($("#epochControl"), state.targetEpoch);
    [els.lambda, els.lr, els.sample, els.noise, els.flip].forEach(setRangeFill);
    generateData();
    if (config.autoStart) { state.running = true; setRunUi(); }
    document.querySelector("#lab").scrollIntoView({ behavior: "smooth" });
  }

  bindControls();
  [els.lambda, els.lr, els.sample, els.noise, els.flip].forEach(setRangeFill);
  generateData();
  requestAnimationFrame(animate);
})();
