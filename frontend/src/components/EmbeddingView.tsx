import { useEffect, useRef } from "react";
import * as THREE from "three";
import type { Modality, Snapshot } from "../lib/types";

const MODALITY_COLORS: Record<Modality, number> = {
  text: 0x9b8cff,
  url: 0x71d3ff,
  pdf: 0xffc266,
  image: 0xff8fb1,
};

interface Props {
  snapshot: Snapshot | null;
  citedIds: Set<string>;
}

export default function EmbeddingView({ snapshot, citedIds }: Props) {
  const mountRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<{
    renderer?: THREE.WebGLRenderer;
    scene?: THREE.Scene;
    camera?: THREE.PerspectiveCamera;
    points?: THREE.Group;
    queryDot?: THREE.Mesh;
    rafId?: number;
    angle: number;
    auto: boolean;
    raycaster: THREE.Raycaster;
    mouse: THREE.Vector2;
    tooltip?: HTMLDivElement;
  }>({ angle: 0, auto: true, raycaster: new THREE.Raycaster(), mouse: new THREE.Vector2() });

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    const mountEl: HTMLDivElement = mount;

    const width = mount.clientWidth;
    const height = mount.clientHeight;

    const scene = new THREE.Scene();
    scene.background = null;

    const camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 1000);
    camera.position.set(0, 1.4, 4.6);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);
    mountEl.appendChild(renderer.domElement);

    // Grid + axes
    const grid = new THREE.GridHelper(8, 16, 0x2c2c44, 0x1d1d2e);
    grid.position.y = -1.2;
    scene.add(grid);

    const axes = new THREE.AxesHelper(0.6);
    axes.position.set(-2.6, -1.18, -2.6);
    scene.add(axes);

    const ambient = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambient);
    const dir = new THREE.DirectionalLight(0xffffff, 0.5);
    dir.position.set(2, 4, 3);
    scene.add(dir);

    const points = new THREE.Group();
    scene.add(points);

    // Query dot
    const queryGeo = new THREE.SphereGeometry(0.07, 24, 24);
    const queryMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const queryDot = new THREE.Mesh(queryGeo, queryMat);
    queryDot.visible = false;
    scene.add(queryDot);

    const queryRingGeo = new THREE.RingGeometry(0.11, 0.14, 32);
    const queryRingMat = new THREE.MeshBasicMaterial({
      color: 0xffffff, side: THREE.DoubleSide, transparent: true, opacity: 0.6,
    });
    const queryRing = new THREE.Mesh(queryRingGeo, queryRingMat);
    queryDot.add(queryRing);

    // Tooltip
    const tooltip = document.createElement("div");
    tooltip.style.cssText =
      "position:absolute;pointer-events:none;background:rgba(20,20,28,0.92);" +
      "color:#eee;font-size:11px;padding:4px 8px;border-radius:6px;" +
      "border:1px solid rgba(255,255,255,0.1);transition:opacity 120ms;" +
      "opacity:0;backdrop-filter:blur(6px);max-width:220px;";
    mountEl.style.position = "relative";
    mountEl.appendChild(tooltip);

    stateRef.current = {
      ...stateRef.current,
      renderer, scene, camera, points, queryDot, tooltip,
    };

    // Drag to orbit
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    let yaw = 0;
    let pitch = 0.25;
    const radius = 5.0;

    function setCameraFromAngles() {
      const x = Math.cos(pitch) * Math.sin(yaw) * radius;
      const z = Math.cos(pitch) * Math.cos(yaw) * radius;
      const y = Math.sin(pitch) * radius;
      camera.position.set(x, y, z);
      camera.lookAt(0, 0, 0);
    }
    setCameraFromAngles();

    function onPointerDown(e: PointerEvent) {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      stateRef.current.auto = false;
    }
    function onPointerMove(e: PointerEvent) {
      const rect = renderer.domElement.getBoundingClientRect();
      stateRef.current.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      stateRef.current.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      if (!dragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      yaw -= dx * 0.005;
      pitch = Math.max(-1.2, Math.min(1.2, pitch + dy * 0.005));
      setCameraFromAngles();
    }
    function onPointerUp() { dragging = false; }
    function onWheel(e: WheelEvent) {
      e.preventDefault();
      const factor = 1 + e.deltaY * 0.0015;
      camera.position.multiplyScalar(factor);
    }

    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    renderer.domElement.addEventListener("wheel", onWheel, { passive: false });

    function tick() {
      if (stateRef.current.auto) {
        yaw += 0.0015;
        setCameraFromAngles();
      }
      // Hover detection
      const st = stateRef.current;
      if (st.points && st.scene) {
        st.raycaster.setFromCamera(st.mouse, camera);
        const hits = st.raycaster.intersectObjects(st.points.children, false);
        const ud = hits[0]?.object?.userData;
        if (ud?.title && tooltip) {
          tooltip.style.opacity = "1";
          tooltip.innerHTML =
            `<div style="font-weight:600">${escapeHtml(ud.title)}</div>` +
            `<div style="opacity:0.7;text-transform:uppercase;font-size:10px;letter-spacing:0.04em">${ud.modality}</div>`;
          const rect = renderer.domElement.getBoundingClientRect();
          const x = ((st.mouse.x + 1) / 2) * rect.width + 12;
          const y = ((1 - st.mouse.y) / 2) * rect.height + 12;
          tooltip.style.left = `${x}px`;
          tooltip.style.top = `${y}px`;
        } else if (tooltip) {
          tooltip.style.opacity = "0";
        }
      }
      renderer.render(scene, camera);
      stateRef.current.rafId = requestAnimationFrame(tick);
    }
    tick();

    function onResize() {
      const w = mountEl.clientWidth;
      const h = mountEl.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    const ro = new ResizeObserver(onResize);
    ro.observe(mountEl);

    return () => {
      ro.disconnect();
      cancelAnimationFrame(stateRef.current.rafId!);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      renderer.domElement.removeEventListener("wheel", onWheel as any);
      mountEl.removeChild(renderer.domElement);
      mountEl.removeChild(tooltip);
      renderer.dispose();
    };
  }, []);

  // Render points whenever snapshot changes
  useEffect(() => {
    const { points, queryDot } = stateRef.current;
    if (!points || !queryDot || !snapshot) return;

    while (points.children.length) {
      const c = points.children.pop()!;
      (c as any).geometry?.dispose?.();
      (c as any).material?.dispose?.();
    }

    if (!snapshot.points || snapshot.points.length === 0) {
      queryDot.visible = false;
      return;
    }

    // Auto-scale into a 3-unit cube around origin.
    const xs = snapshot.points.map((p) => p.x);
    const ys = snapshot.points.map((p) => p.y);
    const zs = snapshot.points.map((p) => p.z);
    if (snapshot.query_point) {
      xs.push(snapshot.query_point.x);
      ys.push(snapshot.query_point.y);
      zs.push(snapshot.query_point.z);
    }
    const span = Math.max(
      Math.max(...xs) - Math.min(...xs),
      Math.max(...ys) - Math.min(...ys),
      Math.max(...zs) - Math.min(...zs),
      0.0001,
    );
    const scale = 2.6 / span;
    const cx = (Math.max(...xs) + Math.min(...xs)) / 2;
    const cy = (Math.max(...ys) + Math.min(...ys)) / 2;
    const cz = (Math.max(...zs) + Math.min(...zs)) / 2;

    function project(p: { x: number; y: number; z: number }) {
      return new THREE.Vector3((p.x - cx) * scale, (p.y - cy) * scale, (p.z - cz) * scale);
    }

    for (const p of snapshot.points) {
      const cited = citedIds.has(p.source_id);
      const color = MODALITY_COLORS[p.modality] ?? 0xaaaaaa;
      const geo = new THREE.SphereGeometry(cited ? 0.07 : 0.045, 18, 18);
      const mat = new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: cited ? 0.6 : 0.15,
        metalness: 0.1,
        roughness: 0.6,
      });
      const sphere = new THREE.Mesh(geo, mat);
      const pos = project(p);
      sphere.position.copy(pos);
      sphere.userData = { title: p.title, modality: p.modality, source_id: p.source_id };
      points.add(sphere);

      if (cited && snapshot.query_point) {
        const q = project(snapshot.query_point);
        const lineGeo = new THREE.BufferGeometry().setFromPoints([pos, q]);
        const lineMat = new THREE.LineBasicMaterial({
          color, transparent: true, opacity: 0.55,
        });
        const line = new THREE.Line(lineGeo, lineMat);
        points.add(line);
      }
    }

    if (snapshot.query_point) {
      const q = project(snapshot.query_point);
      queryDot.position.copy(q);
      queryDot.visible = true;
    } else {
      queryDot.visible = false;
    }
  }, [snapshot, citedIds]);

  function toggleAuto() { stateRef.current.auto = !stateRef.current.auto; }
  function reset() { stateRef.current.auto = true; }

  return (
    <div className="relative h-full w-full">
      <div ref={mountRef} className="w-full h-full" />
      <div className="absolute top-2 right-2 flex gap-1">
        <button
          onClick={toggleAuto}
          className="text-[10px] uppercase tracking-wider px-2 py-1 rounded bg-[color:var(--color-bg-soft)]/80 border border-[color:var(--color-border)] hover:border-[color:var(--color-accent)]/60 backdrop-blur"
        >
          Auto-rotate
        </button>
        <button
          onClick={reset}
          className="text-[10px] uppercase tracking-wider px-2 py-1 rounded bg-[color:var(--color-bg-soft)]/80 border border-[color:var(--color-border)] hover:border-[color:var(--color-accent)]/60 backdrop-blur"
        >
          Reset
        </button>
      </div>
      <Legend />
    </div>
  );
}

function Legend() {
  const items: Array<[Modality, string]> = [
    ["text", "#9b8cff"], ["url", "#71d3ff"], ["pdf", "#ffc266"], ["image", "#ff8fb1"],
  ];
  return (
    <div className="absolute bottom-2 left-2 flex flex-wrap items-center gap-2 px-2 py-1 rounded bg-[color:var(--color-bg-soft)]/80 border border-[color:var(--color-border)] backdrop-blur">
      {items.map(([m, c]) => (
        <span key={m} className="text-[10px] uppercase tracking-wider text-[color:var(--color-fg-muted)] inline-flex items-center gap-1">
          <span className="w-2 h-2 rounded-full" style={{ background: c }} /> {m}
        </span>
      ))}
      <span className="text-[10px] uppercase tracking-wider text-[color:var(--color-fg-muted)] inline-flex items-center gap-1">
        <span className="w-2 h-2 rounded-full bg-white" /> query
      </span>
    </div>
  );
}

function escapeHtml(s: string) {
  return s.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]!));
}
