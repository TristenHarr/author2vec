//! Tab: the personal note, in the author's own words.

use dioxus::prelude::*;

#[component]
pub fn Story() -> Element {
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
                "a little proof attached. I used these famous books and texts, but extrapolate for programming and programmers. "
                "Another reason I made this is because it took a couple prompts while babysitting my other agents. It is so easy to create."
            }
        }
    }
}
