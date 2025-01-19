module Jekyll
    class HighlightLowestValueInRow < Liquid::Block
        def initialize(tag_name, text, tokens)
            super
        end

        def render(context)
            site = context.registers[:site]
            converter = site.find_converter_instance(::Jekyll::Converters::Markdown)
            table_html = converter.convert(super(context))

            table_html.gsub(/<tr>(.*?)<\/tr>/m) do |row|
                # Process the row and get all cells
                row_cells = row.scan(/<td.*?>(.*?)<\/td>/).flatten

                row_values = row_cells.each_with_index
                    .select { |v, idx| idx != row_cells.length - 1 }
                    .map { |v, idx| Float(v) rescue Float::MAX }
                    .compact
                min_value = row_values.min

                # Process the row
                row.gsub(/<td.*?>(.*?)<\/td>/m).with_index do |cell, idx|
                    next cell if idx == row_cells.length - 1
                    # Find the value inside the cell
                    value = Float(cell.match(/<td.*?>(.*?)<\/td>/)[1]) rescue Float::MAX

                    # If this cell is the minimum value, highlight it
                    if value == min_value
                        cell.sub('<td', '<td class="highlight-lowest"')
                    else
                        cell
                    end
                end
            end
        end
    end
end
Liquid::Template.register_tag('highlight_lowest_value_in_row', Jekyll::HighlightLowestValueInRow)