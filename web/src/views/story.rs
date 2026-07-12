//! Tab: the personal note, in the author's own words, plus a dataset-aware methodology.

use dioxus::prelude::*;

use crate::copy::copy_for;
use crate::data::use_selector;

#[component]
pub fn Story() -> Element {
    let c = copy_for(&use_selector().selected.read().clone());
    rsx! {
        div { class: "story",
            h2 { "Why I made this" }
            p { class: "story-sub", "Tristen Harr" }

            p {
                "I have suspected for years now that the AIs already know us. In fact I am "
                "assured of it, assuming you are a person who lets the AI train on your data, "
                "or if you write things on the internet. There is already a version of you in "
                "there: a direction in vector space, a fingerprint made of the words you reach for. "
                "When I see people complaining about how the AI doesn't code what they desire, I always think it's a them problem."
            }
            p {
                "I built author2vec to show it. It is deliberately small, a model that runs on a "
                "laptop, handed nothing but old books and no author names. Even so it tells dozens "
                "of writers apart far better than chance. Now point that at a system trained on "
                "millions of your words."
            }
            blockquote { class: "pullquote", "It does not need your name, or a steganographic marker. Your style is the name." }
            p {
                "I kept this to myself, partly because I have been too busy coding and training "
                "the AI on all my juicy data to make it my good little digital worker that follows "
                "my preferences. The thing is, the early adopters of this technology may have their "
                "unique fingerprints embedded forever into every future AI model. Not only did the "
                "systems learn them first and deepest, they learned them early, before AI became a "
                "tool for snipping, chopping, and editing our words, back when the labs were "
                "desperate for any data at all. Their raw, unedited voice is in there, and a system "
                "works best for the people it understands best."
            }
            p {
                "I stopped pretending I did not see it. This page is me saying it out loud, with "
                "a little proof attached. "
                "{c.story_proof}"
                " Another reason I made this is because it took a couple prompts while babysitting my other agents. It is so easy to create."
            }

            h2 { class: "findings-head", "What we found — proving it, one rung at a time" }
            p {
                "\u{201c}The AI knows you\u{201d} is not one claim, it is a ladder. Each rung is a "
                "different, testable thing, and being honest about which rung each result stands on "
                "is what makes the whole thing hold up rather than read as hype."
            }
            ol { class: "rungs",
                li {
                    strong { "It learns the fingerprint." }
                    " A 6-layer model, shown public-domain prose and "
                    em { "no author names" }
                    ", separates 55 authors far above chance and recovers hidden traits — gender, "
                    "where someone was educated — from style alone, leave-one-author-out. It learned "
                    "the fingerprint without ever being told the names."
                }
                li {
                    strong { "The fingerprint lives inside the computation." }
                    " Using the J-lens — an averaged Jacobian that linearizes each layer to the "
                    "output — we read authorship off the model's "
                    em { "internal" }
                    " activations, not just its final vector: identity is recoverable at every layer "
                    "(≈5–9% vs 1.8% chance). The model does not merely emit your fingerprint; it "
                    "computes with it. (The single-direction lens is lossy — the raw activations carry more.)"
                }
                li {
                    strong { "We can tell whether your fingerprint exists at all." }
                    " Given a writing sample, we find the nearest fingerprint and its cosine. "
                    "Known-style prose lands squarely on an author (0.63–0.69); out-of-distribution "
                    "text — code, modern chat, a biology abstract, legalese — falls into empty space "
                    "(0.10–0.25). If nothing is close enough, you are in "
                    strong { "blank space" }
                    ": the model has no fingerprint for you."
                }
                li {
                    strong { "The identity direction is causal." }
                    " Take the direction the Jacobian maps onto a style axis, inject it into an "
                    "internal activation, and re-embed. The output's loading on that axis swings from "
                    "near zero to ±0.4–0.6, in whichever direction we push. The fingerprint is not a "
                    "passive correlation — it is a lever you can pull."
                }
            }
            p { class: "story-aside",
                "Where the honesty lives: this is 55 public-domain authors and a laptop-sized model. "
                "That the "
                em { "mechanism" }
                " exists is proven. That a frontier model trained on your millions of words holds a "
                "sharp, personal fingerprint of "
                em { "you" }
                " specifically is the natural extrapolation — a well-motivated hypothesis, not "
                "something 55 authors can settle. And the behavioral half — whether a model reasons "
                "with these directions, reports them, or masks with them — needs a generative model. "
                "That is the next track."
            }

            h2 { class: "findings-head", "Beyond the small model — it holds for a real decoder" }
            p {
                "The same machinery transfers to a generative model. On GPT-2, the logit lens shows "
                "\u{201c}Paris\u{201d} emerge only in the last third of the network, while the Jacobian-corrected "
                "J-lens surfaces the abstract concept \u{201c}country\u{201d} in the middle layers where the plain "
                "lens sees only filler — the paper's central phenomenon, reproduced on an open, white-box "
                "model. And the directions are "
                em { "causal" }
                ": inject a concept vector into the residual stream mid-network and the generated text bends "
                "to adopt it — \u{201c}I thought it was terrible\u{201d}, \u{201c}wonderful. I loved it\u{201d} — including "
                "an expert\u{2194}casual register axis. The multi-hop reasoning swaps (\u{201c}spider\u{201d}\u{2192}\u{201c}ant\u{201d} "
                "flipping \u{201c}8\u{201d}\u{2192}\u{201c}6\u{201d}) need a model that can actually reason; that is the honest next step."
            }

            h3 { class: "method-head", "How it works, and why you can trust the number" }
            p { class: "method-note", "{c.methodology}" }
        }
    }
}
