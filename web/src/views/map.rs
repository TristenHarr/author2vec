//! Tab 1 — the 2D vector-space author map, drawn on a canvas with hover tooltips.

use std::collections::HashSet;

use dioxus::prelude::*;
use wasm_bindgen::JsCast;

use super::{gate, Swatch};
use crate::data::{use_dataset, LoadState};
use shared::Bundle;

const CANVAS_ID: &str = "p2v-map";
const W: f64 = 940.0;
const H: f64 = 600.0;
const MARGIN: f64 = 26.0;
/// Squared pixel radius for hover hit-testing.
const HIT_R2: f64 = 110.0;

#[derive(Clone)]
struct Hover {
    idx: usize,
    x: f64,
    y: f64,
}

#[component]
pub fn Map() -> Element {
    let state = use_dataset();
    let mut hidden = use_signal(HashSet::<usize>::new);
    let mut hover = use_signal(|| Option::<Hover>::None);
    let mut screen_pts = use_signal(Vec::<(f64, f64, usize)>::new);

    // Redraw whenever the dataset loads or the author filter changes.
    use_effect(move || {
        if let LoadState::Loaded(bundle) = state() {
            let hide = hidden.read().clone();
            screen_pts.set(draw(&bundle, &hide));
        }
    });

    if let Some(node) = gate(&state()) {
        return node;
    }
    let LoadState::Loaded(bundle) = state() else {
        unreachable!()
    };

    rsx! {
        div { class: "map-view",
            p { class: "explainer",
                "Each dot is one ~200-word passage, placed by PCA of its 384-dimensional embedding "
                "and colored by author. Hollow rings are the held-out \"mystery\" passages. "
                "Hover a dot to read it; click an author in the legend to hide/show them."
            }
            div { class: "map-layout",
                div { class: "map-wrap",
                    canvas {
                        id: CANVAS_ID,
                        onmousemove: move |evt| {
                            let c = evt.element_coordinates();
                            let (mx, my) = (c.x, c.y);
                            let pts = screen_pts.read();
                            let mut best: Option<(f64, usize)> = None;
                            for &(px, py, idx) in pts.iter() {
                                let d = (px - mx).powi(2) + (py - my).powi(2);
                                if best.map_or(true, |(bd, _)| d < bd) {
                                    best = Some((d, idx));
                                }
                            }
                            match best {
                                Some((d, idx)) if d <= HIT_R2 => hover.set(Some(Hover { idx, x: mx, y: my })),
                                _ => hover.set(None),
                            }
                        },
                        onmouseleave: move |_| hover.set(None),
                    }
                    if let Some(h) = hover() {
                        {
                            let p = &bundle.meta.passages[h.idx];
                            let a = &bundle.meta.authors[p.author_id];
                            rsx! {
                                div {
                                    class: "tooltip",
                                    style: "left:{h.x + 14.0}px; top:{h.y + 14.0}px;",
                                    div { class: "tt-author",
                                        Swatch { color: a.color.clone() }
                                        strong { "{a.name}" }
                                        if p.is_mystery {
                                            span { class: "tt-badge", "mystery" }
                                        }
                                    }
                                    div { class: "tt-book", "{p.book_title}" }
                                    p { class: "tt-text", "{snippet(&p.text, 280)}" }
                                }
                            }
                        }
                    }
                }
                div { class: "legend",
                    h3 { "Authors" }
                    for a in bundle.meta.authors.iter() {
                        {
                            let id = a.id;
                            let is_hidden = hidden.read().contains(&id);
                            let count = bundle.meta.passages.iter().filter(|p| p.author_id == id).count();
                            rsx! {
                                button {
                                    key: "{id}",
                                    class: if is_hidden { "legend-item off" } else { "legend-item" },
                                    onclick: move |_| {
                                        let mut h = hidden.write();
                                        if !h.remove(&id) {
                                            h.insert(id);
                                        }
                                    },
                                    Swatch { color: a.color.clone() }
                                    span { class: "legend-name", "{a.name}" }
                                    span { class: "legend-count", "{count}" }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Draw all visible passages and return their pixel coordinates for hit-testing.
fn draw(bundle: &Bundle, hidden: &HashSet<usize>) -> Vec<(f64, f64, usize)> {
    let Some((canvas, ctx)) = context() else {
        return Vec::new();
    };
    let dpr = web_sys::window()
        .map(|w| w.device_pixel_ratio())
        .unwrap_or(1.0)
        .max(1.0);
    canvas.set_width((W * dpr) as u32);
    canvas.set_height((H * dpr) as u32);
    let style = canvas.style();
    let _ = style.set_property("width", &format!("{W}px"));
    let _ = style.set_property("height", &format!("{H}px"));
    let _ = ctx.reset_transform();
    let _ = ctx.scale(dpr, dpr);
    ctx.clear_rect(0.0, 0.0, W, H);

    let (mut minx, mut maxx, mut miny, mut maxy) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for p in &bundle.meta.passages {
        minx = minx.min(p.x);
        maxx = maxx.max(p.x);
        miny = miny.min(p.y);
        maxy = maxy.max(p.y);
    }
    let rx = (maxx - minx).max(1e-6);
    let ry = (maxy - miny).max(1e-6);
    let sx = |x: f32| MARGIN + ((x - minx) / rx) as f64 * (W - 2.0 * MARGIN);
    let sy = |y: f32| H - MARGIN - ((y - miny) / ry) as f64 * (H - 2.0 * MARGIN);

    let mut pts = Vec::with_capacity(bundle.len());
    for (i, p) in bundle.meta.passages.iter().enumerate() {
        if hidden.contains(&p.author_id) {
            continue;
        }
        let (px, py) = (sx(p.x), sy(p.y));
        let color = &bundle.meta.authors[p.author_id].color;
        ctx.begin_path();
        let _ = ctx.arc(
            px,
            py,
            if p.is_mystery { 5.0 } else { 3.0 },
            0.0,
            std::f64::consts::TAU,
        );
        if p.is_mystery {
            ctx.set_line_width(2.0);
            ctx.set_stroke_style_str(color);
            ctx.stroke();
        } else {
            ctx.set_global_alpha(0.72);
            ctx.set_fill_style_str(color);
            ctx.fill();
            ctx.set_global_alpha(1.0);
        }
        pts.push((px, py, i));
    }
    pts
}

fn context() -> Option<(web_sys::HtmlCanvasElement, web_sys::CanvasRenderingContext2d)> {
    let doc = web_sys::window()?.document()?;
    let canvas: web_sys::HtmlCanvasElement = doc.get_element_by_id(CANVAS_ID)?.dyn_into().ok()?;
    let ctx: web_sys::CanvasRenderingContext2d =
        canvas.get_context("2d").ok()??.dyn_into().ok()?;
    Some((canvas, ctx))
}

fn snippet(s: &str, n: usize) -> String {
    let t: String = s.chars().take(n).collect();
    if s.chars().count() > n {
        format!("{t}…")
    } else {
        t
    }
}
