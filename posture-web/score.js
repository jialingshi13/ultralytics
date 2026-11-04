/* Sitting / writing-arm scoring. Ports posture-web/server.py so Free deploys can
   run pose in the browser without PyTorch. Keypoint layout matches COCO-17. */
(function (global) {
  "use strict";

  const NOSE = 0;
  const L_EYE = 1;
  const R_EYE = 2;
  const L_EAR = 3;
  const R_EAR = 4;
  const L_SHOULDER = 5;
  const R_SHOULDER = 6;
  const L_ELBOW = 7;
  const R_ELBOW = 8;
  const L_WRIST = 9;
  const R_WRIST = 10;
  const L_HIP = 11;
  const R_HIP = 12;
  const TRACKED = [
    NOSE, L_EYE, R_EYE, L_EAR, R_EAR,
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP,
  ];
  const ARMS = {
    left: [L_SHOULDER, L_ELBOW, L_WRIST],
    right: [R_SHOULDER, R_ELBOW, R_WRIST],
  };
  const HAND_SWITCH_MARGIN = 0.6;
  const HAND_SWITCH_FRAMES = 8;
  const HOLD_S = 0.3;
  const ARM_HOLD_S = 0.45;
  const GAP_S = 1.0;
  const JUMP_WIDTH = 0.45;
  const ARM_JUMP = 0.15;
  const BONE_LO = 0.7;
  const BONE_HI = 1.35;
  const BONE_WINDOW = 15;
  const BONE_READY = 5;
  const ARM_DEAD_XY = 0.04;
  const ARM_DEAD_DEG = 6;
  const MIN_BONE = 0.18;
  const MIN_EYE = 0.08;
  const CHEST_DROP = 0.42;

  // MediaPipe Pose landmark index -> COCO-17 index.
  const MP_TO_COCO = [
    [0, 0], [2, 1], [5, 2], [7, 3], [8, 4],
    [11, 5], [12, 6], [13, 7], [14, 8], [15, 9], [16, 10], [23, 11], [24, 12],
  ];

  class Euro {
    constructor() {
      this.minCutoff = 1.0;
      this.beta = 0.02;
      this.dCutoff = 1.0;
      this.hat = null;
      this.speed = 0.0;
      this.t = null;
    }

    reset() {
      this.hat = null;
      this.speed = 0.0;
      this.t = null;
    }

    call(t, value) {
      if (this.hat == null || this.t == null) {
        this.hat = value;
        this.speed = 0.0;
        this.t = t;
        return value;
      }
      const dt = t - this.t;
      if (dt <= 1e-6) return this.hat;
      const speed = (value - this.hat) / dt;
      this.speed = this.blend(speed, this.speed, dt, this.dCutoff);
      const cutoff = this.minCutoff + this.beta * Math.abs(this.speed);
      this.hat = this.blend(value, this.hat, dt, cutoff);
      this.t = t;
      return this.hat;
    }

    blend(value, prev, dt, cutoff) {
      const tau = 1.0 / (2.0 * Math.PI * Math.max(cutoff, 1e-6));
      const alpha = 1.0 / (1.0 + tau / dt);
      return alpha * value + (1.0 - alpha) * prev;
    }
  }

  class PointTrack {
    constructor(jumpWidth) {
      this.jumpWidth = jumpWidth == null ? JUMP_WIDTH : jumpWidth;
      this.fx = new Euro();
      this.fy = new Euro();
      this.xy = null;
      this.conf = 0.0;
      this.seen = null;
      this.spike = null;
    }

    reset() {
      this.fx.reset();
      this.fy.reset();
      this.xy = null;
      this.conf = 0.0;
      this.seen = null;
      this.spike = null;
    }

    snapshot() {
      return {
        xy: this.xy ? this.xy.slice() : null,
        conf: this.conf,
        seen: this.seen,
        spike: this.spike ? this.spike.slice() : null,
        fx: { hat: this.fx.hat, speed: this.fx.speed, t: this.fx.t },
        fy: { hat: this.fy.hat, speed: this.fy.speed, t: this.fy.t },
      };
    }

    restore(snap) {
      this.xy = snap.xy ? snap.xy.slice() : null;
      this.conf = snap.conf;
      this.seen = snap.seen;
      this.spike = snap.spike ? snap.spike.slice() : null;
      this.fx.hat = snap.fx.hat;
      this.fx.speed = snap.fx.speed;
      this.fx.t = snap.fx.t;
      this.fy.hat = snap.fy.hat;
      this.fy.speed = snap.fy.speed;
      this.fy.t = snap.fy.t;
    }

    holding(t) {
      return this.xy != null && this.seen != null && t - this.seen <= HOLD_S;
    }

    jumped(x, y, width) {
      if (this.xy == null || width < 1e-6) return false;
      return Math.hypot(x - this.xy[0], y - this.xy[1]) > this.jumpWidth * width;
    }

    confirms(x, y, width) {
      if (this.spike == null || width < 1e-6) return false;
      return Math.hypot(x - this.spike[0], y - this.spike[1]) <= this.jumpWidth * width;
    }

    update(t, x, y, conf, minConf, width) {
      if (conf < minConf) {
        this.spike = null;
        if (this.holding(t)) return [this.xy[0], this.xy[1], this.conf];
        return null;
      }
      if (this.jumped(x, y, width) && !this.confirms(x, y, width)) {
        this.spike = [x, y];
        if (this.holding(t)) return [this.xy[0], this.xy[1], this.conf];
      }
      this.spike = null;
      this.xy = [this.fx.call(t, x), this.fy.call(t, y)];
      this.conf = conf;
      this.seen = t;
      return [this.xy[0], this.xy[1], this.conf];
    }
  }

  function median(values) {
    if (!values.length) return null;
    const ordered = values.slice().sort((a, b) => a - b);
    const mid = Math.floor(ordered.length / 2);
    if (ordered.length % 2) return ordered[mid];
    return (ordered[mid - 1] + ordered[mid]) / 2;
  }

  function boneNear(length, center) {
    if (center == null) return true;
    return length >= BONE_LO * center && length <= BONE_HI * center;
  }

  function pushBone(history, length) {
    history.push(length);
    if (history.length > BONE_WINDOW) history.splice(0, history.length - BONE_WINDOW);
  }

  class PoseTrack {
    constructor() {
      this.points = {};
      for (const index of TRACKED) {
        const armPoint = index === L_ELBOW || index === R_ELBOW || index === L_WRIST || index === R_WRIST;
        this.points[index] = new PointTrack(armPoint ? ARM_JUMP : JUMP_WIDTH);
      }
      this.t = null;
      this.width = 0.0;
      this.pending = null;
      this.streak = 0;
      this.bones = { left: { upper: [], fore: [] }, right: { upper: [], fore: [] } };
      this.armHold = null;
    }

    reset() {
      for (const index of TRACKED) this.points[index].reset();
      this.t = null;
      this.width = 0.0;
      this.pending = null;
      this.streak = 0;
      this.bones = { left: { upper: [], fore: [] }, right: { upper: [], fore: [] } };
      this.armHold = null;
    }

    apply(kpts, minConf, now) {
      if (this.t != null && now - this.t > GAP_S) this.reset();
      this.t = now;
      const out = kpts.map((row) => row.slice());
      const width = this.width;
      const snaps = {};
      for (const index of [L_ELBOW, R_ELBOW, L_WRIST, R_WRIST]) snaps[index] = this.points[index].snapshot();
      for (const index of TRACKED) {
        const [x, y, conf] = kpts[index];
        const smoothed = this.points[index].update(now, x, y, conf, minConf, width);
        if (smoothed == null) out[index][2] = 0.0;
        else {
          out[index][0] = smoothed[0];
          out[index][1] = smoothed[1];
          out[index][2] = smoothed[2];
        }
      }
      this.gateBones(out, minConf, snaps);
      const left = point(out, L_SHOULDER, minConf);
      const right = point(out, R_SHOULDER, minConf);
      if (left && right) {
        const dist = Math.hypot(right[0] - left[0], right[1] - left[1]);
        if (dist >= 1e-6) this.width = dist;
      }
      return out;
    }

    gateBones(out, minConf, snaps) {
      for (const [name, indices] of Object.entries(ARMS)) {
        const [shoulderI, elbowI, wristI] = indices;
        const shoulder = point(out, shoulderI, minConf);
        const elbow = point(out, elbowI, minConf);
        const wrist = point(out, wristI, minConf);
        if (!shoulder || !elbow || !wrist) continue;
        const upper = segment(shoulder, elbow);
        const fore = segment(elbow, wrist);
        const hist = this.bones[name];
        const ready = hist.upper.length >= BONE_READY && hist.fore.length >= BONE_READY;
        const bad = ready && (!boneNear(upper, median(hist.upper)) || !boneNear(fore, median(hist.fore)));
        if (bad && snaps[elbowI].xy && snaps[wristI].xy) {
          this.points[elbowI].restore(snaps[elbowI]);
          this.points[wristI].restore(snaps[wristI]);
          for (const index of [elbowI, wristI]) {
            out[index][0] = snaps[index].xy[0];
            out[index][1] = snaps[index].xy[1];
            out[index][2] = snaps[index].conf;
          }
          continue;
        }
        pushBone(hist.upper, upper);
        pushBone(hist.fore, fore);
      }
    }

    rememberArm(hand, pts) {
      this.armHold = { hand, pts: pts.map((p) => p.slice()), t: this.t };
    }

    heldArm(hand) {
      const hold = this.armHold;
      if (!hold || hold.hand !== hand || this.t == null || hold.t == null) return null;
      if (this.t - hold.t > ARM_HOLD_S) return null;
      return hold.pts;
    }

    lockHand(prevHand, bestName, bestScore, prevScore) {
      if (!prevHand || bestName === prevHand || prevScore == null) {
        this.pending = null;
        this.streak = 0;
        return bestName;
      }
      if (bestScore < prevScore + HAND_SWITCH_MARGIN) {
        this.pending = null;
        this.streak = 0;
        return prevHand;
      }
      if (this.pending === bestName) this.streak += 1;
      else {
        this.pending = bestName;
        this.streak = 1;
      }
      if (this.streak >= HAND_SWITCH_FRAMES) {
        this.pending = null;
        this.streak = 0;
        return bestName;
      }
      return prevHand;
    }
  }

  function point(kpts, index, minConf) {
    const x = kpts[index][0];
    const y = kpts[index][1];
    const conf = kpts[index][2];
    if (conf < minConf) return null;
    return [x, y, conf];
  }

  function jointAngle(a, b, c) {
    const ba = [a[0] - b[0], a[1] - b[1]];
    const bc = [c[0] - b[0], c[1] - b[1]];
    const na = Math.hypot(ba[0], ba[1]);
    const nc = Math.hypot(bc[0], bc[1]);
    if (na < 1e-6 || nc < 1e-6) return null;
    const cos = Math.max(-1, Math.min(1, (ba[0] * bc[0] + ba[1] * bc[1]) / (na * nc)));
    return (Math.acos(cos) * 180) / Math.PI;
  }

  function foldDeg(angle) {
    if (angle > 90) return angle - 180;
    if (angle < -90) return angle + 180;
    return angle;
  }

  function shoulderTilt(left, right) {
    return foldDeg((Math.atan2(right[1] - left[1], left[0] - right[0]) * 180) / Math.PI);
  }

  function torsoLean(shoulderMid, hipMid) {
    return foldDeg((Math.atan2(hipMid[0] - shoulderMid[0], hipMid[1] - shoulderMid[1]) * 180) / Math.PI);
  }

  function bodyAxes(left, right) {
    const span = Math.hypot(right[0] - left[0], right[1] - left[1]);
    if (span < 1e-6) return null;
    const xAxis = [(right[0] - left[0]) / span, (right[1] - left[1]) / span];
    const downA = [-xAxis[1], xAxis[0]];
    const downB = [xAxis[1], -xAxis[0]];
    const yAxis = downA[1] >= downB[1] ? downA : downB;
    return [xAxis, yAxis, span];
  }

  function project(origin, pt, xAxis, yAxis, scale) {
    const dx = pt[0] - origin[0];
    const dy = pt[1] - origin[1];
    return [
      (dx * xAxis[0] + dy * xAxis[1]) / scale,
      (dx * yAxis[0] + dy * yAxis[1]) / scale,
    ];
  }

  function midpoint(a, b) {
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  }

  function segment(a, b) {
    return Math.hypot(a[0] - b[0], a[1] - b[1]);
  }

  function headCenter(leftEar, rightEar, leftEye, rightEye) {
    if (leftEar && rightEar) return midpoint(leftEar, rightEar);
    if (leftEye && rightEye) return midpoint(leftEye, rightEye);
    return null;
  }

  function headPitch(nose, leftEye, rightEye, shoulderWidth) {
    if (shoulderWidth < 1e-6) return null;
    const built = bodyAxes(leftEye, rightEye);
    if (built == null || built[2] < MIN_EYE * shoulderWidth) return null;
    return project(midpoint(leftEye, rightEye), nose, built[0], built[1], built[2])[1];
  }

  function hipsReliable(leftHip, rightHip, shoulderMid, xAxis, yAxis, shoulderWidth, frameW, frameH) {
    if (frameW <= 0 || frameH <= 0 || shoulderWidth < 1e-6) return false;
    const margin = Math.max(2.0, 0.02 * Math.min(frameW, frameH));
    for (const hip of [leftHip, rightHip]) {
      if (hip[0] < margin || hip[1] < margin || hip[0] > frameW - margin || hip[1] > frameH - margin) {
        return false;
      }
    }
    const ratio = segment(leftHip, rightHip) / shoulderWidth;
    if (ratio < 0.55 || ratio > 1.35) return false;
    const below = project(shoulderMid, midpoint(leftHip, rightHip), xAxis, yAxis, shoulderWidth)[1];
    return below >= 0.3;
  }

  function forearmAngle(elbow, wrist, xAxis, yAxis) {
    const dx = wrist[0] - elbow[0];
    const dy = wrist[1] - elbow[1];
    const fx = dx * xAxis[0] + dy * xAxis[1];
    const fy = dx * yAxis[0] + dy * yAxis[1];
    if (Math.hypot(fx, fy) < 1e-6) return null;
    return (Math.atan2(fy, fx) * 180) / Math.PI;
  }

  function angleDelta(a, b) {
    return Math.abs((((a - b + 180) % 360) + 360) % 360 - 180);
  }

  function flatness(angle) {
    let flat = Math.abs(angle);
    if (flat > 90) flat = 180 - flat;
    return flat;
  }

  function writingScore(shoulder, elbow, wrist, xAxis, yAxis) {
    const elbowDeg = jointAngle(shoulder, elbow, wrist);
    const lean = forearmAngle(elbow, wrist, xAxis, yAxis);
    if (elbowDeg == null || lean == null) return null;
    const horizontal = 1.0 - flatness(lean) / 90.0;
    const bent = Math.max(0.0, 1.0 - Math.abs(elbowDeg - 100.0) / 80.0);
    return horizontal + bent;
  }

  function crossPenalty(name, wristX, leftX, rightX) {
    const span = Math.abs(rightX - leftX);
    if (span < 1e-6) return 0.0;
    let crossed;
    if (name === "right") crossed = (wristX - Math.max(leftX, rightX)) / span;
    else crossed = (Math.min(leftX, rightX) - wristX) / span;
    return Math.max(0.0, crossed - 0.15);
  }

  function towardMidline(hand, wristX, targetX) {
    if (hand === "right") return targetX - wristX >= 0.02;
    if (hand === "left") return wristX - targetX >= 0.02;
    return false;
  }

  function emptySit() {
    return {
      status: "unknown",
      shoulder_deg: null,
      torso_deg: null,
      head_ratio: null,
      head_drop: null,
      chest_gap: null,
    };
  }

  function emptyPen() {
    return { status: "unknown", scope: "arm", hand: null, elbow_deg: null, forearm_deg: null };
  }

  function emptyLevels() {
    return { shoulder: "unknown", torso: "unknown", head: "unknown", chest: "unknown", arm: "unknown" };
  }

  function soften(offset, dead) {
    return Math.max(0, offset - dead);
  }

  function bandTol(offset, limit) {
    if (offset <= limit) return "ok";
    if (offset <= limit * 1.2) return "near";
    return "bad";
  }

  function worstLevel(...levels) {
    const rank = { ok: 0, near: 1, bad: 2 };
    const known = levels.filter((level) => level in rank);
    if (!known.length) return "unknown";
    return known.reduce((a, b) => (rank[a] >= rank[b] ? a : b));
  }

  function chooseArm(track, kpts, hand, minConf, prevHand) {
    const found = {};
    for (const [name, indices] of Object.entries(ARMS)) {
      const pts = indices.map((i) => point(kpts, i, minConf));
      if (pts.every(Boolean)) found[name] = pts;
    }
    if (hand in ARMS) {
      track.pending = null;
      track.streak = 0;
      if (hand in found) {
        track.rememberArm(hand, found[hand]);
        return [hand, found[hand]];
      }
      const heldFixed = track.heldArm(hand);
      return heldFixed ? [hand, heldFixed] : [null, null];
    }
    if (!Object.keys(found).length) {
      const heldNone = prevHand && track.heldArm(prevHand);
      return heldNone ? [prevHand, heldNone] : [null, null];
    }

    const leftS = point(kpts, L_SHOULDER, minConf);
    const rightS = point(kpts, R_SHOULDER, minConf);
    let xAxis = [-1.0, 0.0];
    let yAxis = [0.0, 1.0];
    if (leftS && rightS) {
      const built = bodyAxes(leftS, rightS);
      if (built) {
        xAxis = built[0];
        yAxis = built[1];
      }
    }

    const scored = [];
    for (const [name, pts] of Object.entries(found)) {
      let score = writingScore(pts[0], pts[1], pts[2], xAxis, yAxis);
      if (score == null) continue;
      if (leftS && rightS) score -= crossPenalty(name, pts[2][0], leftS[0], rightS[0]);
      if (pts[2][1] > pts[0][1]) score += 0.15;
      scored.push({ name, pts, score });
    }
    if (!scored.length) return [null, null];
    scored.sort((a, b) => b.score - a.score);
    const best = scored[0];
    if (!prevHand) {
      track.pending = null;
      track.streak = 0;
      track.rememberArm(best.name, best.pts);
      return [best.name, best.pts];
    }
    const current = scored.find((item) => item.name === prevHand);
    if (!current) {
      const held = track.heldArm(prevHand);
      if (held) return [prevHand, held];
      track.pending = null;
      track.streak = 0;
      track.rememberArm(best.name, best.pts);
      return [best.name, best.pts];
    }
    const chosen = track.lockHand(prevHand, best.name, best.score, current.score);
    const picked = scored.find((item) => item.name === chosen);
    track.rememberArm(picked.name, picked.pts);
    return [picked.name, picked.pts];
  }

  function judge(track, kpts, limits, prevHand, frameW, frameH) {
    const minConf = limits.kpt_conf;
    const nose = point(kpts, NOSE, minConf);
    const leftEye = point(kpts, L_EYE, minConf);
    const rightEye = point(kpts, R_EYE, minConf);
    const leftEar = point(kpts, L_EAR, minConf);
    const rightEar = point(kpts, R_EAR, minConf);
    const leftShoulder = point(kpts, L_SHOULDER, minConf);
    const rightShoulder = point(kpts, R_SHOULDER, minConf);
    const leftHip = point(kpts, L_HIP, minConf);
    const rightHip = point(kpts, R_HIP, minConf);
    let sit = emptySit();
    const levels = emptyLevels();
    let shoulderWidth = 0.0;
    let xAxis = [-1.0, 0.0];
    let yAxis = [0.0, 1.0];

    if (leftShoulder && rightShoulder) {
      const built = bodyAxes(leftShoulder, rightShoulder);
      if (built) {
        xAxis = built[0];
        yAxis = built[1];
        shoulderWidth = built[2];
        const midS = midpoint(leftShoulder, rightShoulder);
        const shoulderDeg = shoulderTilt(leftShoulder, rightShoulder);
        const center = headCenter(leftEar, rightEar, leftEye, rightEye);
        const headX = center ? project(midS, center, xAxis, yAxis, shoulderWidth)[0] : null;
        let pitch = null;
        if (nose && leftEye && rightEye) pitch = headPitch(nose, leftEye, rightEye, shoulderWidth);
        levels.shoulder = bandTol(Math.abs(shoulderDeg - limits.shoulder_target_deg), limits.shoulder_max_deg);
        const headLevels = [];
        if (headX != null) headLevels.push(bandTol(Math.abs(headX - limits.head_target_x), limits.head_max_ratio));
        if (pitch != null) headLevels.push(bandTol(Math.abs(pitch - limits.head_target_drop), limits.head_drop_tol));
        if (headLevels.length) levels.head = worstLevel(...headLevels);
        let chestGap = null;
        if (nose && shoulderWidth >= 1e-6) {
          const noseY = project(midS, nose, xAxis, yAxis, shoulderWidth)[1];
          chestGap = CHEST_DROP - noseY;
          levels.chest = bandTol(Math.max(0, limits.chest_gap - chestGap), limits.chest_gap_tol);
        }
        let torsoDeg = null;
        if (
          leftHip &&
          rightHip &&
          hipsReliable(leftHip, rightHip, midS, xAxis, yAxis, shoulderWidth, frameW, frameH)
        ) {
          torsoDeg = torsoLean(midS, midpoint(leftHip, rightHip));
          levels.torso = bandTol(Math.abs(torsoDeg - limits.torso_target_deg), limits.torso_max_deg);
        }
        sit = {
          status: worstLevel(levels.shoulder, levels.torso, levels.head, levels.chest),
          shoulder_deg: Math.round(shoulderDeg * 10) / 10,
          torso_deg: torsoDeg == null ? null : Math.round(torsoDeg * 10) / 10,
          head_ratio: headX == null ? null : Math.round(headX * 100) / 100,
          head_drop: pitch == null ? null : Math.round(pitch * 100) / 100,
          chest_gap: chestGap == null ? null : Math.round(chestGap * 100) / 100,
        };
      }
    }

    const [hand, arm] = chooseArm(track, kpts, limits.hand, minConf, prevHand);
    let pen = emptyPen();
    pen.hand = hand;
    if (arm != null) {
      const [shoulder, elbow, wrist] = arm;
      let elbowDeg = jointAngle(shoulder, elbow, wrist);
      let lean = forearmAngle(elbow, wrist, xAxis, yAxis);
      let wristX = null;
      let wristY = null;
      if (shoulderWidth >= 1e-6) {
        if (
          segment(shoulder, elbow) < MIN_BONE * shoulderWidth ||
          segment(elbow, wrist) < MIN_BONE * shoulderWidth
        ) {
          elbowDeg = null;
          lean = null;
        }
        [wristX, wristY] = project(shoulder, wrist, xAxis, yAxis, shoulderWidth);
      }
      const inward = wristX != null && towardMidline(hand, wristX, limits.wrist_target_x);
      if (inward) {
        levels.arm = "ok";
      } else {
        const checks = [];
        if (wristX != null) {
          const dx = wristX - limits.wrist_target_x;
          const dy = wristY - limits.wrist_target_y;
          checks.push(
            bandTol(soften(Math.max(0.0, -dy), ARM_DEAD_XY), limits.wrist_up),
            bandTol(soften(Math.max(0.0, dy), ARM_DEAD_XY), limits.wrist_down),
            bandTol(soften(Math.max(0.0, -dx), ARM_DEAD_XY), limits.wrist_left),
            bandTol(soften(Math.max(0.0, dx), ARM_DEAD_XY), limits.wrist_right),
          );
        }
        if (elbowDeg != null) {
          checks.push(bandTol(soften(Math.abs(elbowDeg - limits.elbow_target_deg), ARM_DEAD_DEG), limits.elbow_tol_deg));
        }
        if (lean != null) {
          checks.push(bandTol(soften(angleDelta(lean, limits.forearm_target_deg), ARM_DEAD_DEG), limits.forearm_tol_deg));
        }
        if (checks.length) levels.arm = worstLevel(...checks);
      }
      pen = {
        status: levels.arm,
        scope: "arm",
        hand,
        elbow_deg: elbowDeg == null ? null : Math.round(elbowDeg * 10) / 10,
        forearm_deg: lean == null ? null : Math.round(lean * 10) / 10,
        wrist_x: wristX == null ? null : Math.round(wristX * 100) / 100,
        wrist_y: wristY == null ? null : Math.round(wristY * 100) / 100,
      };
    }
    return { sit, pen, levels };
  }

  function blank(width, height) {
    return {
      person: false,
      width,
      height,
      keypoints: [],
      sit: emptySit(),
      pen: emptyPen(),
      levels: emptyLevels(),
    };
  }

  function clampNum(value, fallback, lo, hi) {
    const num = Number(value);
    if (!Number.isFinite(num)) return fallback;
    return Math.min(hi, Math.max(lo, num));
  }

  function limitsFromControls(controls) {
    let hand = controls.hand || "auto";
    if (hand !== "left" && hand !== "right" && hand !== "auto") hand = "auto";
    return {
      kpt_conf: clampNum(controls.kpt_conf, 0.5, 0.05, 0.95),
      shoulder_max_deg: clampNum(controls.shoulder_max_deg, 12, 1, 80),
      torso_max_deg: clampNum(controls.torso_max_deg, 15, 1, 80),
      head_max_ratio: clampNum(controls.head_max_ratio, 0.35, 0.02, 1),
      head_drop_tol: clampNum(controls.head_drop_tol, 0.35, 0.02, 1),
      elbow_target_deg: clampNum(controls.elbow_target_deg, 105, 0, 180),
      elbow_tol_deg: clampNum(controls.elbow_tol_deg, 35, 1, 90),
      forearm_target_deg: clampNum(controls.forearm_target_deg, 0, -180, 180),
      forearm_tol_deg: clampNum(controls.forearm_tol_deg, 25, 1, 90),
      shoulder_target_deg: clampNum(controls.shoulder_target_deg, 0, -80, 80),
      torso_target_deg: clampNum(controls.torso_target_deg, 0, -80, 80),
      head_target_x: clampNum(controls.head_target_x, 0, -1, 1),
      head_target_drop: clampNum(controls.head_target_drop, 0.7, -0.2, 2),
      chest_gap: clampNum(controls.chest_gap, 0.9, 0.2, 1.8),
      chest_gap_tol: clampNum(controls.chest_gap_tol, 0.12, 0.02, 0.6),
      wrist_target_x: clampNum(controls.wrist_target_x, 0, -2, 2.5),
      wrist_target_y: clampNum(controls.wrist_target_y, 0, -2, 2.5),
      wrist_up: clampNum(controls.wrist_up, 0, 0, 2.5),
      wrist_down: clampNum(controls.wrist_down, 1.5, 0, 2.5),
      wrist_left: clampNum(controls.wrist_left, 1, 0, 2.5),
      wrist_right: clampNum(controls.wrist_right, 1, 0, 2.5),
      hand,
    };
  }

  function cocoFromMediaPipe(landmarks, width, height) {
    const kpts = Array.from({ length: 17 }, () => [0, 0, 0]);
    if (!landmarks || !landmarks.length) return null;
    for (const [mp, coco] of MP_TO_COCO) {
      const lm = landmarks[mp];
      if (!lm) continue;
      const conf = typeof lm.visibility === "number" ? lm.visibility : 1;
      kpts[coco] = [lm.x * width, lm.y * height, conf];
    }
    return kpts;
  }

  function scoreFrame(track, landmarks, width, height, controls, prevHand, nowSec) {
    const empty = blank(width, height);
    const raw = cocoFromMediaPipe(landmarks, width, height);
    if (!raw) return empty;
    const limits = limitsFromControls(controls);
    const kpts = track.apply(raw, limits.kpt_conf, nowSec);
    const left = point(kpts, L_SHOULDER, limits.kpt_conf);
    const right = point(kpts, R_SHOULDER, limits.kpt_conf);
    if (!left || !right) return empty;
    const payload = blank(width, height);
    payload.person = true;
    payload.keypoints = kpts.map((row) => row.map((v) => Math.round(v * 10) / 10));
    Object.assign(payload, judge(track, kpts, limits, prevHand || null, width, height));
    return payload;
  }

  global.PostureScore = {
    PoseTrack,
    blank,
    limitsFromControls,
    cocoFromMediaPipe,
    scoreFrame,
    towardMidline,
  };
})(typeof window !== "undefined" ? window : globalThis);
