(() => {
  const trees = document.querySelectorAll("canvas.growth");
  const stillness = matchMedia("(prefers-reduced-motion: reduce)");

  // Branches the shader holds: the stem a tree stands on, a grown tree's thirteen, and the three the next tree has
  // while the view dives.
  const AT_MOST = 24;
  const LEAVES = 7;
  // A beat is one split, as silvic grows trees: the leaf with the most gain splits next.
  const A_BEAT = 0.9;
  const TO_GROW = 0.45;
  // A grown tree stands for a beat, then the view dives into its topmost leaf, from which the next tree has
  // sprouted, a fifth the size. Every tree grows from a leaf of the one before, without end.
  const STANDING = 1;
  const DIVING = 1.8;
  const DIVE_AT = LEAVES - 1 + TO_GROW + STANDING;
  const NEARER = 5;
  const GROUND = [0, -0.72, 0];
  // During the dive the rest of the crown falls away (over these fractions of the dive) while the branches leading
  // to the leaf stay, narrowing to the stem every tree stands on: the new tree arrives framed as the first one was,
  // still joined to the branch it grew from.
  const FALL_FROM = 0.15;
  const FALL_OVER = 0.55;
  const NARROW_FROM = 0.3;
  const NARROW_OVER = 0.7;
  const STEM_GIRTH = 0.17;
  const ROOTS = { from: [0, GROUND[1] - 3, 0], to: GROUND, girth: STEM_GIRTH, bud: 0 };
  const STAYING = { turn: [1, 0, 0, 0, 1, 0, 0, 0, 1], scale: 1, shift: [0, 0, 0] };

  for (const canvas of trees) {
    const glass = canvas.getContext("webgl", clearly());
    const everywhere = glass && heard(glass, glass.VERTEX_SHADER, where_everything_is());
    const a_moment = glass && heard(glass, glass.FRAGMENT_SHADER, what_a_moment_looks_like());
    const grove = everywhere && a_moment && held_together(glass, everywhere, a_moment);
    if (!grove) { keep_still(canvas); continue; }

    laid_down(glass, grove, one_triangle_wide_enough_to_hold_it_all());
    const at = (name) => glass.getUniformLocation(grove, name);

    const room = four_to_six_times(canvas.clientWidth || 22);
    canvas.width = canvas.height = room;
    glass.viewport(0, 0, room, room);
    glass.uniform2f(at("room"), room, room);
    tinted(glass, at);

    // The tree in view, in its own coordinates, which are the world's; the branches of the tree it grew from that
    // lead to it; the tree growing from its topmost leaf while the view dives.
    let generation = 0;
    let here = { tree: sown(0), beats: 0 };
    let behind = null;
    let ahead = null;
    let swaying = 0;
    let pace = 1;
    let wish = 1;
    let then = lately();
    let waiting = 0;

    const be = () => {
      const diving = ahead ? smooth(Math.min(1, Math.max(0, (here.beats - DIVE_AT) / DIVING))) : 0;
      const view = ahead ? dived(ahead, diving) : STAYING;
      shown(glass, at, gathered(here, behind, ahead, diving), view, swaying);
      glass.drawArrays(glass.TRIANGLES, 0, 3);
    };

    const pass = (when) => {
      const a_little = Math.min((when - then) / 1000, 0.1);
      then = when;
      pace += (wish - pace) * Math.min(1, a_little * 6);
      here.beats += (a_little * pace) / A_BEAT;
      swaying += a_little * pace;
      if (!ahead && here.beats >= DIVE_AT) ahead = planned(here.tree, generation + 1);
      if (ahead && here.beats >= DIVE_AT + DIVING) {
        behind = { tree: here.tree, place: undone(ahead.place), leading: ahead.leading, narrowed: ahead.narrowed };
        here = { tree: ahead.tree, beats: here.beats - DIVE_AT };
        ahead = null;
        generation++;
      }
      be();
      waiting = wait_for(pass);
    };

    const go = () => {
      forget(waiting);
      if (stillness.matches) {
        here = { tree: sown(0), beats: DIVE_AT - STANDING / 2 };
        behind = ahead = null;
        be();
        return;
      }
      be();
      then = lately();
      waiting = wait_for(pass);
    };

    const a_hand = canvas.closest("a") ?? canvas.parentElement;
    a_hand?.addEventListener("pointerenter", () => (wish = 3));
    a_hand?.addEventListener("pointerleave", () => (wish = 1));

    go();
    stillness.addEventListener("change", go);
  }

  // One tree, every branch with the beat it sprouts on. Leaf-wise, like silvic: each beat the leaf with the largest
  // gain splits in two, so long, open limbs branch first and the crown fills in after.
  function sown(generation) {
    const chance = seeded(0x51b1c + generation * 7919);
    const trunk = {
      from: GROUND,
      toward: [0, 1, 0],
      length: 0.3,
      girth: 0.15,
      depth: 0,
      born: 0,
      gain: 1,
      parent: null,
      children: [],
    };
    const branches = [trunk];
    for (let beat = 1; beat < LEAVES; beat++) {
      const best = leaves_of(branches).reduce((a, b) => (b.gain > a.gain ? b : a));
      const tip = along(best.from, best.toward, best.length);
      for (const side of [1, -1]) {
        const length = best.length * (best.depth === 0 ? 0.95 : 0.74) * (0.92 + 0.16 * chance());
        const child = {
          from: tip,
          toward: bent(best.toward, 0.55 + 0.3 * chance(), side, (chance() - 0.5) * 0.7),
          length,
          girth: Math.max(0.07, best.girth * 0.8),
          depth: best.depth + 1,
          born: beat,
          gain: length * (0.5 + 0.5 * chance()) * (1 - 0.12 * (best.depth + 1)),
          parent: best,
          children: [],
        };
        best.children.push(child);
        branches.push(child);
      }
    }
    return branches;
  }

  function leaves_of(branches) {
    return branches.filter((b) => !b.children.length);
  }

  // The next tree grows from the grown tree's topmost leaf, where the crown leaves the most room: its ground sits on
  // the leaf's tip, a fifth the size, along the leaf. The dive is the one similarity that maps the next tree's
  // coordinates onto this one's, run a fraction at a time about its fixed point, so the view spirals in on the leaf
  // rather than sliding to it.
  function planned(tree, generation) {
    const leaves = leaves_of(tree);
    const tip_of = (b) => along(b.from, b.toward, b.length);
    const leaf = leaves.reduce((a, b) => (tip_of(b)[1] > tip_of(a)[1] ? b : a));
    const leading = new Set();
    for (let branch = leaf; branch; branch = branch.parent) leading.add(branch);
    const tip = tip_of(leaf);
    const axis = cross([0, 1, 0], leaf.toward);
    const sine = Math.hypot(...axis);
    const angle = Math.atan2(sine, leaf.toward[1]);
    const about = sine > 1e-6 ? axis.map((x) => x / sine) : [0, 0, 1];
    const turn = rotation(about, angle);
    const scale = 1 / NEARER;
    const shift = minus(tip, times(applied(turn, GROUND), scale));
    const still_point = solved(minus_matrix(STAYING.turn, scaled_matrix(turn, scale)), shift);
    // The leaf narrows until, seen from the next tree, it is the stem: as thick as the roots the first tree has.
    const narrowed = STEM_GIRTH / (leaf.girth * NEARER);
    return { tree: sown(generation), leaf, leading, narrowed, place: { turn, scale, shift }, about, angle, still_point };
  }

  function dived(ahead, diving) {
    const scale = Math.pow(1 / NEARER, diving);
    const turn = rotation(ahead.about, ahead.angle * diving);
    const shift = minus(ahead.still_point, times(applied(turn, ahead.still_point), scale));
    return { turn, scale, shift };
  }

  // The branches to draw, in world coordinates: what the tree in view stands on (the first tree's roots, or the
  // branches that led to it), the tree in view, and during a dive the tree sprouting from its leaf. Branches that
  // do not lead to that leaf fall away as the view dives; the ones that do narrow into the next tree's stem.
  function gathered(here, behind, ahead, diving) {
    const standing_on = behind
      ? laid_out(behind.tree, Infinity, behind.place, (b) => (behind.leading.has(b) ? [behind.narrowed, 0] : null))
      : [ROOTS];
    if (!ahead) return [...standing_on, ...laid_out(here.tree, here.beats, STAYING, () => [1, 1])];
    const falling = 1 - smooth(Math.min(1, Math.max(0, (diving - FALL_FROM) / FALL_OVER)));
    const thinning = 1 + (ahead.narrowed - 1) * smooth(Math.min(1, Math.max(0, (diving - NARROW_FROM) / NARROW_OVER)));
    const sprouting = here.beats - DIVE_AT;
    const giving_way = 1 - eased(Math.min(1, Math.max(0, sprouting / TO_GROW)));
    const stem = standing_on.map((limb) => ({ ...limb, girth: limb.girth * thinning }));
    const grown = laid_out(here.tree, here.beats, STAYING, (b) => {
      if (!ahead.leading.has(b)) return [falling, falling];
      return [thinning, b === ahead.leaf ? giving_way : 1];
    });
    const sprout = laid_out(ahead.tree, sprouting, ahead.place, () => [1, 1]);
    return [...stem, ...grown, ...sprout].filter((limb) => limb.girth > 1e-3).slice(0, AT_MOST);
  }

  // `kept(branch)` scales a branch's girth and bud (`[girth, bud]`), or leaves it out (`null`).
  function laid_out(tree, beats, place, kept) {
    const limbs = [];
    for (const branch of tree) {
      if (branch.born > beats) continue;
      const scales = kept(branch);
      if (!scales) continue;
      const grown = eased(Math.min(1, (beats - branch.born) / TO_GROW));
      const tip = along(branch.from, branch.toward, branch.length * grown);
      const sprouted = branch.children.filter((child) => child.born <= beats);
      const outgrown = sprouted.length ? eased(Math.min(1, (beats - sprouted[0].born) / TO_GROW)) : 0;
      limbs.push({
        from: placed(place, branch.from),
        to: placed(place, tip),
        girth: branch.girth * scales[0] * place.scale,
        bud: 0.12 * grown * (1 - outgrown) * scales[1] * place.scale,
      });
    }
    return limbs;
  }

  function shown(glass, at, limbs, view, swaying) {
    const from = new Float32Array(AT_MOST * 4);
    const to = new Float32Array(AT_MOST * 4);
    limbs.forEach((limb, i) => {
      from.set([...limb.from, limb.girth], i * 4);
      to.set([...limb.to, limb.bud], i * 4);
    });
    const m = view.turn.map((x) => x * view.scale);
    glass.uniform4fv(at("branch_from"), from);
    glass.uniform4fv(at("branch_to"), to);
    glass.uniform1f(at("count"), limbs.length);
    glass.uniformMatrix3fv(at("view"), false, new Float32Array([m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]));
    glass.uniform3fv(at("view_shift"), view.shift);
    glass.uniform1f(at("view_scale"), view.scale);
    glass.uniform1f(at("turning"), 0.45 * Math.sin(swaying * 0.7));
  }

  function tinted(glass, at) {
    const style = getComputedStyle(document.documentElement);
    for (const [uniform, token] of [["dusk", "--green-below"], ["day", "--green"], ["dawn", "--green-above"]]) {
      glass.uniform3fv(at(uniform), as_light(style.getPropertyValue(token)));
    }
  }

  function as_light(hex) {
    const digits = hex.trim().replace("#", "");
    return [0, 2, 4].map((i) => parseInt(digits.slice(i, i + 2), 16) / 255);
  }

  function along(from, toward, length) {
    return from.map((x, i) => x + toward[i] * length);
  }

  // `toward` tipped by `angle` to one `side` within the plane that faces the viewer, give or take `depth`, then
  // drawn a little back toward the sky: the crown keeps a tree's silhouette while it sways.
  function bent(toward, angle, side, depth) {
    const flat = cross(toward, [0, 0, 1]);
    const aside = unit([flat[0] * side, flat[1] * side, flat[2] + depth]);
    const turned = toward.map((x, i) => x * Math.cos(angle) + aside[i] * Math.sin(angle));
    return unit([turned[0], turned[1] + 0.15, turned[2]]);
  }

  // A similarity `x -> scale * turn * x + shift`; `turn` is a row-major rotation.
  function placed(place, x) {
    return applied(place.turn, x).map((v, i) => v * place.scale + place.shift[i]);
  }

  function undone(place) {
    const turn = transposed(place.turn);
    const scale = 1 / place.scale;
    return { turn, scale, shift: times(applied(turn, place.shift), -scale) };
  }

  // Rodrigues: a turn by `angle` about the unit `axis`.
  function rotation([x, y, z], angle) {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const k = 1 - c;
    return [
      c + x * x * k, x * y * k - z * s, x * z * k + y * s,
      y * x * k + z * s, c + y * y * k, y * z * k - x * s,
      z * x * k - y * s, z * y * k + x * s, c + z * z * k,
    ];
  }

  function applied(m, v) {
    return [
      m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
      m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
      m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ];
  }

  function transposed(m) {
    return [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
  }

  function minus_matrix(a, b) {
    return a.map((x, i) => x - b[i]);
  }

  function scaled_matrix(m, s) {
    return m.map((x) => x * s);
  }

  // `x` with `m x = v`, by Cramer's rule.
  function solved(m, v) {
    const det = (a) =>
      a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
    const whole = det(m);
    return [0, 1, 2].map((column) => {
      const swapped = m.slice();
      for (let row = 0; row < 3; row++) swapped[row * 3 + column] = v[row];
      return det(swapped) / whole;
    });
  }

  function cross(a, b) {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  }

  function minus(a, b) {
    return a.map((x, i) => x - b[i]);
  }

  function times(v, s) {
    return v.map((x) => x * s);
  }

  function unit(v) {
    const length = Math.hypot(...v);
    return v.map((x) => x / length);
  }

  function eased(t) {
    return 1 - Math.pow(1 - t, 3);
  }

  function smooth(t) {
    return t * t * (3 - 2 * t);
  }

  // mulberry32: the same tree every time a generation number comes around.
  function seeded(seed) {
    let state = seed >>> 0;
    return () => {
      state = (state + 0x6d2b79f5) >>> 0;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function clearly() {
    return { alpha: true, antialias: false, premultipliedAlpha: true };
  }

  function keep_still(canvas) {
    canvas.classList.add("still");
  }

  function heard(glass, kind, words) {
    const said = glass.createShader(kind);
    glass.shaderSource(said, words);
    glass.compileShader(said);
    if (glass.getShaderParameter(said, glass.COMPILE_STATUS)) return said;
    console.error(glass.getShaderInfoLog(said));
    return null;
  }

  function held_together(glass, ...voices) {
    const grove = glass.createProgram();
    for (const voice of voices) glass.attachShader(grove, voice);
    glass.linkProgram(grove);
    if (!glass.getProgramParameter(grove, glass.LINK_STATUS)) {
      console.error(glass.getProgramInfoLog(grove));
      return null;
    }
    glass.useProgram(grove);
    return grove;
  }

  function laid_down(glass, grove, shape) {
    glass.bindBuffer(glass.ARRAY_BUFFER, glass.createBuffer());
    glass.bufferData(glass.ARRAY_BUFFER, shape, glass.STATIC_DRAW);
    const place = glass.getAttribLocation(grove, "place");
    glass.enableVertexAttribArray(place);
    glass.vertexAttribPointer(place, 2, glass.FLOAT, false, 0, 0);
  }

  function four_to_six_times(shown) {
    return Math.ceil(shown * Math.min(6, Math.max(4, (devicePixelRatio || 1) * 2)));
  }

  function one_triangle_wide_enough_to_hold_it_all() {
    return new Float32Array([-1, -1, 3, -1, -1, 3]);
  }

  function lately() {
    return performance.now();
  }

  function wait_for(next) {
    return requestAnimationFrame(next);
  }

  function forget(waiting) {
    cancelAnimationFrame(waiting);
  }

  function where_everything_is() {
    return `
      attribute vec2 place;
      void main() { gl_Position = vec4(place, 0.0, 1.0); }`;
  }

  function what_a_moment_looks_like() {
    return `
      precision highp float;
      uniform vec2 room;
      uniform float turning;
      uniform float count;
      uniform vec4 branch_from[${AT_MOST}];
      uniform vec4 branch_to[${AT_MOST}];
      uniform mat3 view;
      uniform vec3 view_shift;
      uniform float view_scale;
      uniform vec3 dusk;
      uniform vec3 day;
      uniform vec3 dawn;

      float together(float a, float b, float k) {
        float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
        return mix(b, a, h) - k * h * (1.0 - h);
      }

      mat2 turn(float a) {
        float c = cos(a), s = sin(a);
        return mat2(c, s, -s, c);
      }

      float limb(vec3 p, vec3 from, vec3 to, float girth) {
        vec3 along = p - from, all = to - from;
        float h = clamp(dot(along, all) / max(dot(all, all), 1e-9), 0.0, 1.0);
        return length(along - all * h) - girth * mix(1.0, 0.85, h);
      }

      // Joints blend in proportion to the branches, so a tree a fifth the size is the same tree.
      float wood(vec3 p) {
        float here = 1e9;
        for (int i = 0; i < ${AT_MOST}; i++) {
          if (float(i) >= count) break;
          vec4 from = branch_from[i];
          vec4 to = branch_to[i];
          float d = limb(p, from.xyz, to.xyz, from.w);
          if (to.w > 0.0) d = together(d, length(p - to.xyz) - to.w, to.w * 0.25);
          here = together(here, d, from.w * 0.3);
        }
        return here;
      }

      // The stem every tree stands on ends just above the bottom of the view, rounded off rather than cut by the
      // frame: the wood is met with the half-space above \`resting\` by a smooth maximum.
      const float resting = -0.86;

      float tree(vec3 p) {
        float above = resting - p.y;
        p.xz *= turn(turning);
        float d = wood(view * p + view_shift) / view_scale;
        return -together(-d, -above, 0.1);
      }

      vec3 facing(vec3 p) {
        vec2 e = vec2(0.0015, 0.0);
        return normalize(vec3(
          tree(p + e.xyy) - tree(p - e.xyy),
          tree(p + e.yxy) - tree(p - e.yxy),
          tree(p + e.yyx) - tree(p - e.yyx)));
      }

      void main() {
        vec2 seen = (2.0 * gl_FragCoord.xy - room) / room.y;
        vec3 eye = vec3(0.0, -0.16, 2.5);
        vec3 look = normalize(vec3(seen * 0.6, -2.0));

        float travelled = 0.0, nearest = 1e9, met = -1.0;
        for (int i = 0; i < 90; i++) {
          float d = tree(eye + look * travelled);
          nearest = min(nearest, d);
          if (d < 0.002) { met = travelled; break; }
          travelled += d * 0.85;
          if (travelled > 5.0) break;
        }

        vec3 colour;
        float there;
        if (met >= 0.0) {
          vec3 p = eye + look * met;
          vec3 n = facing(p);
          vec3 sun = normalize(vec3(0.5, 0.85, 0.7));
          float lit = clamp(dot(n, sun), 0.0, 1.0);
          float edge = pow(1.0 - clamp(dot(n, -look), 0.0, 1.0), 3.0);
          colour = mix(dusk, day, 0.2 + 0.8 * lit);
          colour += edge * dawn * 0.6;
          colour += pow(clamp(dot(reflect(-sun, n), -look), 0.0, 1.0), 44.0) * 0.9;
          colour += pow(clamp(dot(n, normalize(vec3(-0.4, -0.6, 0.5))), 0.0, 1.0), 8.0) * dusk * 0.5;
          there = 1.0;
        } else {
          colour = day;
          there = 1.0 - smoothstep(0.002, max(0.003, 1.6 / room.y), nearest);
        }
        gl_FragColor = vec4(colour * there, there);
      }`;
  }
})();
