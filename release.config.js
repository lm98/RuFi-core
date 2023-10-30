const config = require('semantic-release-preconfigured-conventional-commits')
const verifyReleaseCommands = `
sed -i 's/version.*/version = "\${nextRelease.version}"/g' Cargo.toml || exit 1
git add -A || exit 2
git commit -m "chore: [skip ci] update version in Cargo.toml" || exit 3
cargo publish --dry-run || exit 4
`
const prepareCommands = `
sed -i 's/version.*/version = "\${nextRelease.version}"/g' Cargo.toml || exit 1
git add -A || exit 2
git commit -m "chore: [skip ci] update version in Cargo.toml" || exit 3
git push --force origin || exit 4
`
const publishCommands = `
git tag -a -f \${nextRelease.version} \${nextRelease.version} -F CHANGELOG.md || exit 1
git push --force origin \${nextRelease.version} || exit 2
cargo package || exit 3
cargo publish || exit 4
`
const releaseBranches = ["main"]
config.branches = releaseBranches
config.plugins.push(
    ["@semantic-release/exec", {
        "prepareCmd": prepareCommands,
        "publishCmd": publishCommands,
        "verifyReleaseCmd": verifyReleaseCommands,
    }],
    ["@semantic-release/github", {
        "assets": [
            { "path": "target/package/*.crate" },
        ]
    }],
    ["@semantic-release/git", {
        "assets": ["CHANGELOG.md", "package.json"],
        "message": "chore(release)!: [skip ci] ${nextRelease.version} released"
    }],
)
module.exports = config