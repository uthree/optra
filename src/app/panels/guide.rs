//! The guides, readable inside the application.
//!
//! The release criterion is a fresh install taken to working tracking using
//! only what the application itself provides, and a markdown file in a
//! repository is not that: the person mid-setup has the app open and no reason
//! to know the repository exists. The same three guides that ship in `doc/`
//! are compiled in and rendered here, so they cannot be lost, stale relative
//! to the binary, or blocked by a machine with no browser signed into
//! anything.

use egui_commonmark::{CommonMarkCache, CommonMarkViewer};

use super::PanelContext;

/// One guide: its tab label, the file name links refer to it by, and its text.
struct Section {
    title: &'static str,
    file: &'static str,
    text: &'static str,
}

/// The guides, in the order a first install meets them.
const SECTIONS: [Section; 3] = [
    Section {
        title: "Placing the cameras",
        file: "cameras.md",
        text: include_str!("../../../doc/cameras.md"),
    },
    Section {
        title: "Calibrating a room",
        file: "calibration.md",
        text: include_str!("../../../doc/calibration.md"),
    },
    Section {
        title: "Troubleshooting",
        file: "troubleshooting.md",
        text: include_str!("../../../doc/troubleshooting.md"),
    },
];

#[derive(Default)]
pub struct GuidePanel {
    selected: usize,
    cache: CommonMarkCache,
}

impl GuidePanel {
    pub fn ui(&mut self, ui: &mut egui::Ui, _ctx: &mut PanelContext<'_>) {
        ui.horizontal(|ui| {
            for (index, section) in SECTIONS.iter().enumerate() {
                ui.selectable_value(&mut self.selected, index, section.title);
            }
        });

        ui.separator();

        let section = &SECTIONS[self.selected];
        // One scroll area per guide, so switching back to a guide returns to
        // where the reader left it rather than to wherever the other one was.
        egui::ScrollArea::vertical()
            .id_salt(section.file)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                // Prose reads badly at full window width; cap it at something
                // like a page and let a narrow window wrap below that.
                ui.set_max_width(720.0);
                CommonMarkViewer::new().show(ui, &mut self.cache, section.text);
            });

        // The guides link to each other by file name. Rendered as a hyperlink,
        // a click would be handed to the OS as a URL it cannot open; caught
        // here, it switches the tab, which is where that file actually is.
        ui.ctx().output_mut(|output| {
            output.commands.retain(|command| {
                let egui::OutputCommand::OpenUrl(open) = command else {
                    return true;
                };
                match SECTIONS.iter().position(|section| section.file == open.url) {
                    Some(index) => {
                        self.selected = index;
                        false
                    }
                    None => true,
                }
            });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::SECTIONS;

    /// Every link in the guides has to go somewhere. A link to a section is
    /// caught and switches the tab; anything else would be handed to the OS as
    /// a URL, and a relative file name is not one. This is what stops a future
    /// guide from linking to design.md, which does not ship in the panel.
    #[test]
    fn every_link_in_the_guides_lands_on_a_section() {
        for section in &SECTIONS {
            let mut rest = section.text;
            while let Some(at) = rest.find("](") {
                rest = &rest[at + 2..];
                let end = rest.find(')').expect("an unclosed link");
                let target = &rest[..end];

                assert!(
                    SECTIONS.iter().any(|section| section.file == target),
                    "{} links to '{target}', which is not a guide in this panel",
                    section.file
                );
            }
        }
    }

    /// The tab labels are how a user finds a guide, and an empty guide is a
    /// packaging accident that would otherwise ship silently.
    #[test]
    fn no_guide_is_empty_or_unnamed() {
        for section in &SECTIONS {
            assert!(!section.title.is_empty());
            assert!(
                section.text.len() > 500,
                "{} is implausibly short",
                section.file
            );
        }
    }
}
