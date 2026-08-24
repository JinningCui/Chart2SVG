const { optimize } = require('svgo');
const fs = require('fs');
const path = require('path');
const fsp = fs.promises;

/**
 * 单个 SVG/XML 优化逻辑
 * 使用 SVGO 工业级插件进行扁平化和精简，并删除位图图像
 */
async function optimizeSingleFile(content) {
    const result = optimize(content, {
        multipass: process.env.SVGO_MULTIPASS !== 'false',
        plugins: [
            'preset-default', 
            'convertStyleToAttrs', 
            // 自定义插件：删除所有的 <image> 标签（位图/Base64）
            {
                name: 'removeBitmapImages',
                fn: () => {
                    return {
                        element: {
                            enter: (node, parentNode) => {
                                if (node.name === 'image') {
                                    parentNode.children = parentNode.children.filter(child => child !== node);
                                }
                            },
                        },
                    };
                },
            },
            {
                name: 'convertTransform',
                params: {
                    collapseIntoPaths: true, 
                    // Small map transforms (for example 0.0858) must not be
                    // rounded to zero, otherwise an entire geographic layer
                    // disappears.
                    floatPrecision: 3,
                }
            },
            {
                name: 'cleanupNumericValues',
                params: {
                    floatPrecision: 1,
                }
            },
            {
                name: 'removeViewBox',
                active: false 
            }
        ],
    });
    return result.data;
}

/**
 * 预处理并修复常见的 XML/SVG 结构问题
 */
function preprocessSVGContent(content) {
    const startTag = '<svg';
    const endTag = '</svg>';
    
    const startIndex = content.indexOf(startTag);
    if (startIndex === -1) return null;

    let depth = 0;
    let currentIndex = startIndex;
    let matchedEndIndex = -1;

    while (currentIndex < content.length) {
        const nextOpen = content.indexOf(startTag, currentIndex);
        const nextClose = content.indexOf(endTag, currentIndex);

        if (nextClose === -1) break; 

        if (nextOpen !== -1 && nextOpen < nextClose) {
            depth++;
            currentIndex = nextOpen + startTag.length;
        } else {
            depth--;
            if (depth === 0) {
                matchedEndIndex = nextClose + endTag.length;
                break;
            }
            currentIndex = nextClose + endTag.length;
        }
    }

    if (matchedEndIndex === -1) return null;

    let svg = content.slice(startIndex, matchedEndIndex).trim();

    if (/xlink:/i.test(svg)) {
        const openingTagEnd = svg.indexOf('>');
        const openingTag = svg.slice(0, openingTagEnd);
        
        if (!/xmlns:xlink/i.test(openingTag)) {
            svg = svg.slice(0, 4) + ' xmlns:xlink="http://www.w3.org/1999/xlink"' + svg.slice(4);
        }
    }
    
    return svg;
}

/**
 * 并行批量处理器
 */
async function processBeagleSources(sourceDirs, outputBase) {
    if (!fs.existsSync(outputBase)) {
        await fsp.mkdir(outputBase, { recursive: true });
    }

    let overallStats = {
        totalRawLength: 0,
        totalOptimizedLength: 0,
        processedCount: 0,
        skippedCount: 0
    };

    const CONCURRENCY_LIMIT = 20; 

    for (const sourceDir of sourceDirs) {
        let stats = {
            totalRawLength: 0,
            totalOptimizedLength: 0,
            processedCount: 0,
            skippedCount: 0
        };
        if (!fs.existsSync(sourceDir)) {
            console.warn(`\n⚠️ [跳过] 路径未找到: ${sourceDir}`);
            continue;
        }

        const normalizedSource = path.resolve(sourceDir);
        const relativePath = path.relative(path.resolve(outputBase), normalizedSource);

        const outputRoot = path.join(outputBase, relativePath);
        let folderNames = fs.readdirSync(sourceDir).filter(name => 
            fs.statSync(path.join(sourceDir, name)).isDirectory()
        );

        const selectedIds = new Set(
            (process.env.SVGO_IDS || '').split(',').map(name => name.trim()).filter(Boolean)
        );
        if (selectedIds.size > 0) {
            folderNames = folderNames.filter(name => selectedIds.has(name));
        }

        const shardCount = Number.parseInt(process.env.SVGO_SHARD_COUNT || '1', 10);
        const shardIndex = Number.parseInt(process.env.SVGO_SHARD_INDEX || '0', 10);
        if (shardCount > 1) {
            const stableHash = (name) => Array.from(name).reduce(
                (hash, char) => ((hash * 31) + char.charCodeAt(0)) >>> 0,
                0
            );
            folderNames = folderNames.filter(name => stableHash(name) % shardCount === shardIndex);
        }
        const startIndex = Number.parseInt(process.env.SVGO_START_INDEX || '0', 10);
        if (startIndex > 0) {
            folderNames = folderNames.slice(startIndex);
        }

        console.log(`\n📂 数据集: ${relativePath}`);
        console.log(`待处理文件夹总数: ${folderNames.length}`);

        for (let i = 0; i < folderNames.length; i += CONCURRENCY_LIMIT) {
            const chunk = folderNames.slice(i, i + CONCURRENCY_LIMIT);
            
            await Promise.all(chunk.map(async (folderName) => {
                const svgPath = path.join(sourceDir, folderName, 'cleaned_svg.txt');
                if (!fs.existsSync(svgPath)) {
                    console.warn(`\n⚠️ 跳过 [文件夹: ${folderName}] 在 [${relativePath}]`);
                    console.warn(`   原因: 缺少 cleaned_svg.txt`);
                    stats.skippedCount++;
                    return;
                }

                try {
                    const fileContent = await fsp.readFile(svgPath, 'utf8');
                    const rawContent = preprocessSVGContent(fileContent);
                    
                    if (!rawContent) {
                        console.warn(`\n⚠️ 跳过 [文件夹: ${folderName}] 在 [${relativePath}]`);
                        console.warn(`   原因: 未匹配到完整 <svg>...</svg>`);
                        stats.skippedCount++;
                        return;
                    }

                    const maxOptimizeBytes = Number.parseInt(
                        process.env.SVGO_MAX_BYTES || '0', 10
                    );
                    const preserveRendering = /data-preserve-rendering=["']true["']/i.test(rawContent);
                    const optimizedSVG = preserveRendering || (maxOptimizeBytes > 0 && rawContent.length > maxOptimizeBytes)
                        ? rawContent
                        : await optimizeSingleFile(rawContent);
                    const outputDir = path.join(outputRoot, folderName);
                    
                    if (!fs.existsSync(outputDir)) {
                        await fsp.mkdir(outputDir, { recursive: true });
                    }

                    const outputPath = path.join(outputDir, 'svg.txt');
                    await fsp.writeFile(outputPath, optimizedSVG);

                    stats.totalRawLength += rawContent.length;
                    stats.totalOptimizedLength += optimizedSVG.length;
                    stats.processedCount += 1;

                } catch (err) {
                    console.error(`\n❌ 错误 [文件夹: ${folderName}] 在 [${relativePath}]`);
                    console.error(`   原因: ${err.message}`);
                    stats.skippedCount++;
                }
            }));

            if (i % (CONCURRENCY_LIMIT * 5) === 0 || i + CONCURRENCY_LIMIT >= folderNames.length) {
                const progress = (((i + chunk.length) / folderNames.length) * 100).toFixed(1);
                process.stdout.write(`\r进度: ${progress}% (${i + chunk.length}/${folderNames.length}) | 已处理: ${stats.processedCount} `);
            }
        }
        console.log(`\n✅ 数据集完成: ${relativePath}`);

        if (stats.processedCount > 0) {
            const avgLength = stats.totalOptimizedLength / stats.processedCount;
            const reduction = ((stats.totalRawLength - stats.totalOptimizedLength) / stats.totalRawLength) * 100;
            console.log('--------------------------------------------------');
            console.log(`📊 数据集统计: ${relativePath}`);
            console.log(`- 处理成功文件数:  ${stats.processedCount}`);
            console.log(`- 跳过/失败文件数: ${stats.skippedCount}`);
            console.log(`- 平均 SVG 长度:   ${avgLength.toFixed(2)} 字符`);
            console.log(`- 总长度压缩率:    ${reduction.toFixed(2)}%`);
            console.log('--------------------------------------------------');
        } else {
            console.log('--------------------------------------------------');
            console.log(`📊 数据集统计: ${relativePath}`);
            console.log(`- 处理成功文件数:  0`);
            console.log(`- 跳过/失败文件数: ${stats.skippedCount}`);
            console.log('--------------------------------------------------');
        }

        overallStats.totalRawLength += stats.totalRawLength;
        overallStats.totalOptimizedLength += stats.totalOptimizedLength;
        overallStats.processedCount += stats.processedCount;
        overallStats.skippedCount += stats.skippedCount;
    }

    if (overallStats.processedCount > 0) {
        const avgLength = overallStats.totalOptimizedLength / overallStats.processedCount;
        const reduction = ((overallStats.totalRawLength - overallStats.totalOptimizedLength) / overallStats.totalRawLength) * 100;

        console.log('\n==================================================');
        console.log('📊 清洗统计结果:');
        console.log(`- 处理成功文件数:  ${overallStats.processedCount}`);
        console.log(`- 跳过/失败文件数: ${overallStats.skippedCount}`);
        console.log(`- 平均 SVG 长度:   ${avgLength.toFixed(2)} 字符`);
        console.log(`- 总长度压缩率:    ${reduction.toFixed(2)}%`);
        
        const csvContent = `metric,value\nprocessed_count,${overallStats.processedCount}\navg_length,${avgLength.toFixed(2)}\nreduction_percent,${reduction.toFixed(2)}`;
        const shardCount = Number.parseInt(process.env.SVGO_SHARD_COUNT || '1', 10);
        const shardIndex = Number.parseInt(process.env.SVGO_SHARD_INDEX || '0', 10);
        const summaryName = shardCount > 1 ? `svgo_summary_shard_${shardIndex}.csv` : 'svgo_summary.csv';
        await fsp.writeFile(path.join(outputBase, summaryName), csvContent);
        console.log(`- 统计报表已保存:  ${path.join(outputBase, summaryName)}`);
        console.log('==================================================');
    }
}

// --- 路径配置 ---
const outputBase = path.join(__dirname, 'data', 'Beagle');
const allDatasetNames = ['chartblocks', 'fusion_clean', 'graphiq_clean', 'plotly_export', 'echarts'];
const selectedDatasetNames = process.env.SVGO_DATASETS
    ? process.env.SVGO_DATASETS.split(',').map(name => name.trim()).filter(Boolean)
    : allDatasetNames;
const sourceDirs = selectedDatasetNames.map(name => path.join(outputBase, name, 'charts'));

processBeagleSources(sourceDirs, outputBase)
    .then(() => console.log('\n✨ 所有任务已顺利执行完毕！'))
    .catch(console.error);
